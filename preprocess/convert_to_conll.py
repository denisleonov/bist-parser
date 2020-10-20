import numpy as np
import pickle
import json
import utils
import re
import argparse

import spacy
from spacy_wordnet.wordnet_annotator import WordnetAnnotator 


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input', dest='input_path', 
						default='data/region_graphs.p',
						help='Required Processed input file', 
						metavar='FILE')
	parser.add_argument('-o', '--output', dest='output_path', 
						default='coco_train.conll',
						help='Processed file output file path', 
						metavar='FILE')
	parser.add_argument('-t', '--train', dest='is_training', 
						default=True,
						help='Check if processed file required Training')
	args = parser.parse_args()

	with open(args.input_path, 'rb') as f:
		region_graphs = pickle.load(f)
	
	nlp = spacy.load('en')
	nlp.add_pipe(WordnetAnnotator(nlp.lang), after='tagger')
	
	node_list = get_nodes(region_graphs, nlp)

	with open(args.output_path, 'w') as fout:
		for node in node_list:		
			fout.write(str(node.id))
			fout.write('\t' + node.word)
			fout.write('\t' + (str(node.parent_id) if node.parent_id is not None else '_')) 
			fout.write('\t' + (str(node.rel) if node.rel is not None else '_'))
			fout.write('\t' + (str(node.prop) if node.prop is not None else '_') + '\n')
		fout.write('\n')
		
	print('Finish Alignment :)')

class Node:
	def __init__(self, idx):
		self.id        = idx
		self.parent_id = None
		self.rel       = None
		self.prop      = None
		self.word      = None
		self.synsets   = None

def find(word: str, phrase_words: list[str]) -> int:
	sentence = ' ' + ' '.join(phrase_words) + ' '
	# find exactly word, not subword
	return sentence.find(' ' + word + ' ')

def find_object(obj_name: str, phrase_words: list[str]) -> tuple[bool, list[int]]:
	find_id = find(obj_name, phrase_words)
	if find_id == -1:
		return False, []
	else:
		sentence = ' ' + ' '.join(phrase_words) + ' '
		# replace whole name with TEMPTOK
		temp = sentence.replace(' ' + obj_name + ' ', ' TEMPTOK ')
		temp = temp.split()
		# find id for the first (head) word in the object name (name can be a combination of words)
		head_id = temp.index('TEMPTOK')
		
		# if a obj_name is one word: True, [head_id]
		# if it's combination of words: True, [head_id, head_id + 1, ..., head_id + num_words_in_name]
		num_words_in_name = len(obj_name.split())
		return (
			True, 
		    [word_id for word_id 
			 in range(head_id, head_id + num_words_in_name)]
		)

def find_pos(phrase_sentt: str, word: str) -> int:
	phrase_sentt = ' ' + phrase_sentt + ' '
	temp = phrase_sentt.replace(' ' + word + ' ', ' TEMPTOK ')
	temp = temp.split()
	
	return temp.index('TEMPTOK') + len(word.split()) - 1

def find_wordnet(word: str, node_list: list[Node], nlp: WordnetAnnotator) -> bool, int:
	# if combination of words
	if len(word.split()) > 1:
		word = '_'.join(word.split())

	word_wn = utils.word_to_wordnet(word, nlp)
	max_id = 0
	max_lap = 0

	for node_id in range(len(node_list)):
		if node_list[word_id].prop != None:
			continue

		overlap = word_wn.intersection(node_list[word_id].synsets)
		if len(overlap) > max_lap:
			max_lap = len(overlap)
			max_id = node_id
			#return True, node_id # TODO: ????????????!!!!!!!!!!!!!!!!
    
	return (True, max_id) if max_lap > 0 else (False, None)

def find_pos_wordnet(node_list, word, prop, nlp): #especially for finding objects
	if len(word.split()) > 1:
		word = '_'.join(word.split())
	word_synsets = utils.word_to_wordnet(word, nlp)

	max_lap = 0
	max_id = 0

	for node_id in range(len(node_list)):
		overlap = word_synsets.intersection(node_list[node_id].synsets)
		if len(overlap) > max_lap:
			max_lap = len(overlap)
			max_id  = node_id
	
	return max_id if max_lap > 0 else None


def lower_tuples(tuples, prop='words'):
	if prop == 'words':
		return [word.lower() for word in tuples]

	elif prop == 'attributes':
		return[
			[attr_pair[0].lower(), [a.lower() for a in attr_pair[1]]] 
		    for attr_pair in tuples
		]

	else:
		return[
			[rels_pair[0].lower(), rels_pair[1].lower(), rels_pair[2].lower()] 
			for rels_pair in tuples
		]
def get_nodes(region_graphs, nlp):
	for graph_id in range(len(region_graphs)):
		conll = dict()
		vocab = []
		vocab_to_id = dict()
		node_list = []
		obj_set = set()

		input_sent = re.sub('"', ' ', ' '.join(region_graphs[graph_id].phrase.lower().split()))
		input_sent = ' '.join(input_sent.split())
		phrase_sent = ' ' + input_sent + ' '
		phrase     = lower_tuples(input_sent.split(), 0)
		objects    = lower_tuples(region_graphs[graph_id].objects, 'words')
		attributes = lower_tuples(region_graphs[graph_id].attributes, 'attributes')
		relations  = lower_tuples(region_graphs[graph_id].relationships, 'relationships')
		
		phrase_words = phrase[:]

		for word_id in range(len(phrase)):
			node         = Node(word_id+1)
			node.word    = phrase[word_id]
			node.synsets = utils.word_to_wordnet(node.word)
			node_list.append(node)

			if phrase[word_id] in vocab_to_id:
				vocab_to_id[phrase[word_id]].append(word_id)
			else:
				vocab_to_id[phrase[word_id]] = [word_id]
		
		# type(obj) = str
		for obj in objects:
			(is_exist, id_list) = find_object(obj, phrase_words)
			if is_exist:
				obj_set.add(' '.join(obj.split())) # object name can be a combination of words
				# set SAME relationship for whole combination of words
				for word_idx in range(len(id_list)):
					if word_idx != len(id_list) - 1:
						node_list[id_list[word_idx]].parent_id = node_list[id_list[word_idx + 1]].id
						node_list[id_list[word_idx]].rel       = 'same'

					node_list[id_list[word_idx]].prop = "OBJ"
					phrase_words[id_list[word_idx]] = "OBJ_" + str(word_idx)

		for attr_pair in attributes:
			# if phrase contains current object
			if attr_pair[0] in obj_set:
				found_idx = find_pos(phrase_sent, attr_pair[0])
			else:
				continue

			for attr in attr_pair[1]:
				if (find(phrase_words, attr) + 1):
					attr_tail_id = find_pos(' '.join(phrase_words), attr)

					node_list[attr_tail_id].parent_id = node_list[found_idx].id
					node_list[attr_tail_id].rel  = "ATTR"
					node_list[attr_tail_id].prop = "ATTR"
					for attr_id in xrange(len(attr.split())-1, 0, -1):
						node_list[attr_tail_id - attr_id].parent_id = node_list[attr_tail_id - attr_id +1].id
						node_list[attr_tail_id - attr_id].rel       = 'same'
						phrase_words[attr_tail_id - attr_id] = 'ATTR_%d' % (len(attr.split()) - attr_id - 1)

					phrase_words[attr_tail_id] = 'ATTR_%d' % (len(attr.split()) - 1)

				else:
					(isATTR, idx) = find_wordnet(attr, node_list, nlp)
					if isATTR:
						node_list[idx].parent_id = node_list[found_idx].id
						node_list[idx].rel  = "ATTR"
						node_list[idx].prop = "ATTR"
						phrase_words[idx] = "ATTR_0"

		for rel_pair in relations:
			sub  = ' '.join(rel_pair[0].split())
			obj  = ' '.join(rel_pair[2].split())
			pred = rel_pair[1]
			if (sub not in obj_set) or (obj not in obj_set):
				continue
			
			sub_idx = find_pos(phrase_sent, sub)
			obj_idx = find_pos(phrase_sent, obj)
			
			if (find(phrase_words, pred)+1):
				pred_tail_id = find_pos(phrase_sent, pred)
				if node_list[pred_tail_id].prop != None and node_list[pred_tail_id].prop != "PRED":
					continue
				for pred_id in xrange(len(pred.split())-1,0,-1):
					node_list[pred_tail_id - pred_id].parent_id = node_list[pred_tail_id - pred_id +1].id
					node_list[pred_tail_id - pred_id].rel       = 'same'
			else:
				(isPred ,pred_tail_id) = find_wordnet(rel_pair[1], node_list, nlp)
				if not isPred:
					continue
			
			node_list[pred_tail_id].prop = "PRED"
			phrase_words[pred_tail_id] = "PRED"
			
			node_list[pred_tail_id].rel = "PRED"
			node_list[obj_idx].rel = "OBJT"

			node_list[pred_tail_id].parent_id = node_list[sub_idx].id
			node_list[obj_idx].parent_id = node_list[pred_tail_id].id

		#print "phrase_words before second object:  ", phrase_words
		for obj in objects:
			if obj in obj_set:
				continue
			else:
				(isObj, idx) = find_wordnet(obj, node_list, nlp)
				if isObj:
					if node_list[idx].prop != None:
						continue
					phrase_words[idx] = "OBJ_000"
					node_list[idx].prop = "OBJ"
					
					obj_set.add(obj)

		for attr_pair in attributes:

			if (find(phrase_sent.split(), attr_pair[0])+1):
				found_idx = find_pos(phrase_sent, attr_pair[0])

			else:
				found_idx = find_pos_wordnet(node_list, attr_pair[0], "OBJ", nlp)
				if not isinstance(found_idx, int):
					continue

			for attr in attr_pair[1]:
				if (find(phrase_words, attr)+1):
					attr_tail_id = find_pos(' '.join(phrase_words), attr)
					node_list[attr_tail_id].parent_id = node_list[found_idx].id
					node_list[attr_tail_id].rel  = "ATTR"
					node_list[attr_tail_id].prop = "ATTR"
					for attr_id in xrange(len(attr.split())-1, 0, -1):
						node_list[attr_tail_id - attr_id].parent_id = node_list[attr_tail_id - attr_id +1].id
						node_list[attr_tail_id - attr_id].rel       = 'same'
						phrase_words[attr_tail_id - attr_id] = 'ATTR_%d' % (len(attr.split()) - attr_id - 1)

					phrase_words[attr_tail_id] = 'ATTR_%d' % (len(attr.split()) - 1)

				else:
					(isATTR, idx) = find_wordnet(attr, node_list, nlp)
					if isATTR:
						node_list[idx].parent_id = node_list[found_idx].id
						node_list[idx].rel  = "ATTR"
						node_list[idx].prop = "ATTR"
						phrase_words[idx] = "ATTR_0"


		for rel_pair in relations:
			sub  = ' '.join(rel_pair[0].split())
			obj  = ' '.join(rel_pair[2].split())
			pred = rel_pair[1]
			if (sub not in obj_set) or (obj not in obj_set):
				continue

			if (find(phrase_sent.split(), sub)+1):

				sub_idx = find_pos(phrase_sent, sub)
			else:
				sub_idx = find_pos_wordnet(node_list, sub, "OBJ", nlp)
				if not isinstance(sub_idx, int):
					print "error subjs= word: ", sub 
					print sub
					print obj_set
					print phrase_sent
					print phrase_words
					print node_list[1].prop
					print node_list[0].prop
					print attributes
					print objects
					print relations
					print utils.similar(node_list[2].synsets, utils.word_to_wn(sub))
					exit()
			if (find(phrase_sent.split(), obj)+1):
				obj_idx = find_pos(phrase_sent, obj)
			else:
				obj_idx = find_pos_wordnet(node_list, obj, "OBJ", nlp)
				if not isinstance(obj_idx, int):
					print "error objs= word: ", obj 
					print obj
					print obj_set
					print phrase_sent
					print phrase_words
					print node_list[1].prop
					print relations
					print objects
					print utils.similar(node_list[1].synsets, utils.word_to_wn(sub))
					exit()
			
			if (find(phrase_words, pred)+1):
				pred_tail_id = find_pos(phrase_sent, pred)
				if node_list[pred_tail_id].prop != None and node_list[pred_tail_id].prop != "PRED":
					continue
				for pred_id in xrange(len(pred.split())-1,0,-1):
					node_list[pred_tail_id - pred_id].parent_id = node_list[pred_tail_id - pred_id +1].id
					node_list[pred_tail_id - pred_id].rel       = 'same'
			else:
				(isPred ,pred_tail_id) = find_wordnet(rel_pair[1], node_list, nlp)
				if not isPred:
					continue
			
			node_list[pred_tail_id].prop = "PRED"
			phrase_words[pred_tail_id] = "PRED"
			
			node_list[pred_tail_id].rel = "PRED"
			node_list[obj_idx].rel = "OBJT"

			node_list[pred_tail_id].parent_id = node_list[sub_idx].id
			node_list[obj_idx].parent_id = node_list[pred_tail_id].id

		isDuplicate = False
		for node in node_list:
			if node.prop == 'OBJ' and node.parent_id == None:
				node.parent_id = 0
			if TRAIN:
				if node.parent_id == node.id:  #turn off when generate dev and test conll
					isDuplicate = True
					break

		if isDuplicate:
			continue
