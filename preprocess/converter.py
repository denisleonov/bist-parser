import pickle
import re
import spacy

from collections import namedtuple
from spacy_wordnet.wordnet_annotator import WordnetAnnotator
from tqdm import tqdm

from visual_genome_parsing import Graph

def remove_extra_spaces(string):
    return ' '.join(string.split())

ProcessedGraph = namedtuple('ParsedGraph', 
                            ['phrase_sent',
                             'phrase_words',
                             'objects',
                             'attributes',
                             'relations'])

class Node:
    def __init__(self, idx):
        self.id        = idx
        self.parent_id = None
        self.rel       = None
        self.prop      = None
        self.word      = None
        self.synsets   = None

class ToConllConverter:
    def __init__(self, 
                 region_graphs,
                 output_file='data/coco_train.conll'):
        self.region_graphs = region_graphs
        self.output_file = output_file

        self.nlp = spacy.load('en')
        self.nlp.add_pipe(WordnetAnnotator(self.nlp.lang), after='tagger')
        
    def word_to_wordnet(self, word):
        token = self.nlp(word)[0]
        return set([l.name() for l in token._.wordnet.lemmas()])

    def process_tuples(self, tuples, prop='words'):
        if prop == 'words':
            return [remove_extra_spaces(word.lower()) for word in tuples]

        elif prop == 'attributes':
            return[
                (remove_extra_spaces(attr_pair[0].lower()), 
                 [remove_extra_spaces(a.lower()) for a in attr_pair[1]]) 
                for attr_pair in tuples
            ]

        else:
            return[
                [remove_extra_spaces(r.lower()) for r in rels_pair] 
                for rels_pair in tuples
            ]

    def find(self, word, phrase_words):
        sentence = ' ' + ' '.join(phrase_words) + ' '
        # find exactly word, not subword
        return sentence.find(' ' + word + ' ')

    def find_word_by_word(self, word, phrase_words):
        find_id = self.find(word, phrase_words)
        if find_id == -1:
            return []
        
        sentence = ' ' + ' '.join(phrase_words) + ' '
        
        # replace whole name with TEMPTOK
        temp = sentence.replace(' ' + word + ' ', ' TEMPTOK ')
        temp = temp.split()
        
        # find id for the first (head) word in the object name (name can be a combination of words)
        head_id = temp.index('TEMPTOK')
        
        # if a word is one word: True, [head_id]
        # if it's combination of words: True, [head_id, head_id + 1, ..., head_id + num_words_in_name]
        num_words_in_name = len(word.split())

        return [word_id for word_id 
                in range(head_id, head_id + num_words_in_name)]

    def find_by_synonyms(self, word, nodes):
        # if combination of words
        if len(word.split()) > 1:
            word = '_'.join(word.split())

        obj_synsets = self.word_to_wordnet(word)
        max_id = 0
        max_lap = 0

        for node_id, node in enumerate(nodes):
            # if the node is already in use
            if node.prop != None:
                continue
            
            # search for the most similar word in phrase by synonyms overlap
            overlap = obj_synsets.intersection(node.synsets)
            if len(overlap) > max_lap:
                max_lap = len(overlap)
                max_id = node_id
        
        return max_id if max_lap > 0 else None

    def initialize_nodes(self, phrase_words):
        node_list = []
        for word_id, word in enumerate(phrase_words):
            node         = Node(word_id + 1)
            node.word    = word
            node.synsets = self.word_to_wordnet(word)
            node_list.append(node)

        return node_list

    def process_graph(self, graph):
        # use split and join to remove extra spaces
        phrase_sent = re.sub('"', ' ', ' '.join(graph.phrase.lower().split()))
        phrase_sent = ' ' + ' '.join(phrase_sent.split()) + ' '

        phrase_words = phrase_sent.split()
        objects    = self.process_tuples(graph.objects, 'words')
        attributes = self.process_tuples(graph.attributes, 'attributes')
        relations  = self.process_tuples(graph.relationships, 'relationships')
        
        return ProcessedGraph(phrase_sent, phrase_words, objects, attributes, relations)

    def align_objects_nodes(self, nodes, obj_to_id, graph_objects, phrase_words, is_first_pass=True):
        if is_first_pass:
            for obj in graph_objects:
                obj_ids = self.find_word_by_word(obj, phrase_words)
                if obj_ids:
                    # if object is found then
                    # remove extra spaces cause object name could be a combination of words
                    obj = ' '.join(obj.split())
                    # and set last found id
                    obj_to_id[obj] = obj_ids[-1]
                
                # set SAME relationship for whole combination of words
                for word_id, node_id in enumerate(obj_ids):
                    if word_id != len(obj_ids) - 1:
                        nodes[node_id].parent_id = nodes[node_id + 1].id
                        nodes[node_id].rel       = 'same'

                    nodes[node_id].prop = 'OBJ'

            return nodes, obj_to_id
        
        # if is_first_pass == False then use synonym match
        for obj in graph_objects:
            # if the object has already been found with word by word match then skip it
            if obj in obj_to_id:
                continue
            # else use synonym match 
            else:
                node_id = self.find_by_synonyms(obj, nodes)
                if node_id is not None:
                    assert nodes[node_id].prop == None
                    nodes[node_id].prop = 'OBJ'
                    obj_to_id[obj] = node_id

        return nodes, obj_to_id

    def align_attributes_nodes(self, nodes, obj_to_id, attributes, phrase_words):
        for attr_pair in attributes:
            # if phrase contains current object
            if attr_pair[0] in obj_to_id:
                obj_id = obj_to_id[attr_pair[0]]
            else:
                continue

            for attr in attr_pair[1]:
                # if find using word by word match
                attr_ids = self.find_word_by_word(attr, phrase_words)
                if attr_ids:
                    attr_tail_id = attr_ids[-1]
                    
                    for word_id, node_id in enumerate(attr_ids):
                        if word_id != len(attr_ids) - 1:
                            nodes[node_id].parent_id = nodes[node_id + 1].id
                            nodes[node_id].rel       = 'same'

                # else use synonym match
                else:
                    attr_tail_id = self.find_by_synonyms(attr, nodes)
                    if attr_tail_id is None:
                        continue
                
                nodes[attr_tail_id].parent_id = nodes[obj_id].id
                nodes[attr_tail_id].rel  = 'ATTR'
                nodes[attr_tail_id].prop = 'ATTR'

        return nodes

    def align_relations_nodes(self, nodes, obj_to_id, relations, phrase_words):
        for rel_pair in relations:
            sub_name, pred, obj_name = rel_pair
            if (sub_name not in obj_to_id) or (obj_name not in obj_to_id):
                continue
            
            sub_id = obj_to_id[sub_name]
            obj_id = obj_to_id[obj_name]
            
            # if find using word by word match
            rel_ids = self.find_word_by_word(pred, phrase_words)
            if rel_ids:
                pred_tail_id = rel_ids[-1]

                if nodes[pred_tail_id].prop is not None and nodes[pred_tail_id].prop != 'PRED':
                    continue

                for word_id, node_id in enumerate(rel_ids):
                    if word_id != len(rel_ids) - 1:
                        nodes[node_id].parent_id = nodes[node_id + 1].id
                        nodes[node_id].rel       = 'same'
            # else use synonym match
            else:
                pred_tail_id = self.find_by_synonyms(pred, nodes)
                if pred_tail_id is None:
                    continue
            
            nodes[pred_tail_id].prop = 'PRED'			
            nodes[pred_tail_id].rel = 'PRED'
            nodes[obj_id].rel = 'OBJT'

            nodes[pred_tail_id].parent_id = nodes[sub_id].id
            nodes[obj_id].parent_id = nodes[pred_tail_id].id

        return nodes

    def build_conll_nodes(self):
        pbar = tqdm(enumerate(self.region_graphs), total=len(self.region_graphs))
        for ind, graph in pbar:
            graph = self.process_graph(graph)
            nodes = self.initialize_nodes(graph.phrase_words)
            
            obj_to_id = {}
            is_first_pass = True
            for step in range(2):
               # print('####################')
               #print(len(nodes))
                nodes, obj_to_id = self.align_objects_nodes(nodes,  
                                                            obj_to_id,
                                                            graph.objects, 
                                                            graph.phrase_words, 
                                                            is_first_pass=is_first_pass)
                #print(len(nodes))
                nodes = self.align_attributes_nodes(nodes,
                                                    obj_to_id, 
                                                    graph.attributes, 
                                                    graph.phrase_words)
                #print(len(nodes))
                nodes = self.align_relations_nodes(nodes,
                                                   obj_to_id, 
                                                   graph.relations, 
                                                   graph.phrase_words)
                is_first_pass = False
                #pbar.set_postfix_str(f'graph #{ind}; step {step + 1}/2')

            with open(self.output_file, 'a+') as fout:
                for n in nodes:		
                    fout.write(str(n.id))
                    fout.write('\t' + n.word)
                    fout.write('\t' + (str(n.parent_id) if n.parent_id is not None else '_')) 
                    fout.write('\t' + (str(n.rel) if n.rel is not None else '_'))
                    fout.write('\t' + (str(n.prop) if n.prop is not None else '_') + '\n')
                fout.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input', dest='input_path', 
						default='data/region_graphs.p',
						help='Required Processed input file', 
						metavar='FILE')
	parser.add_argument('-o', '--output', dest='output_path', 
						default='data/coco_train.conll',
						help='Processed file output file path', 
						metavar='FILE')
	parser.add_argument('-t', '--train', dest='is_training', 
						default=True,
						help='Check if processed file required Training')
	args = parser.parse_args()

	with open(args.input_path, 'rb') as f:
		region_graphs = pickle.load(f)
    
    converter = ToConllConverter(region_graphs, args.output_path)
    converter.build_conll_nodes()

