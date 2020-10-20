import ijson
import multiprocessing as mp
import pickle

from tqdm import tqdm

class Graph:
	def __init__(self, 
	             phrase, 
				 objects,
				 attributes,
				 relationships):
		self.phrase = phrase
		self.objects = objects
		self.attributes = attributes
		self.relationships = relationships

def process_vg_data(path2image_data='data/image_data.json',
					path2region_graphs='data/region_graphs.json',
					path2attributes='data/attributes.json',
					use_only_coco=True):
	global pbar

	imgs = open(path2image_data)
	iter_imgs = ijson.items(imgs, 'item')

	graphs = open(path2region_graphs)
	iter_graphs = ijson.items(graphs, 'item')

	attrs = open(path2attributes)
	iter_attrs = ijson.items(attrs, 'item')

	with mp.Pool(mp.cpu_count()) as pool:
		for i, (img, graph, attr) in enumerate(zip(iter_imgs, iter_graphs, iter_attrs)):
			# check if image exists in the MS COCO
			if use_only_coco and img['coco_id'] == None:
				pbar.update()
				continue				
            
			pool.apply_async(parse_region_graph,
			                 args=(i, img, graph, attr), 
							 callback=collect_async_results)

def collect_async_results(result):
	global region_graphs, pbar
	region_graphs.extend(result)
	pbar.set_postfix_str(f'processed region graphs: {len(region_graphs)}')
	pbar.update()

def parse_region_graph(index, img, graph, attr):
	global region_graphs

	regions = graph['regions']
	attributes = attr['attributes']
	
	obj_to_attr = {}
	for a in attributes:
		obj_id = a['object_id']
		try:
			obj_to_attr[obj_id] = a['attributes']
		except:
			continue
    
	image_graphs = []
	for r in regions:
		# if no phrase for a region
		if not len(r['phrase'].strip().split()):
			continue

		if len(r['objects']) == 0:
			continue

		if len(region_graphs):
			# if same phrase
			if region_graphs[-1][1].phrase == r['phrase']:
				continue

		r_objects = []
		r_attributes = []
		r_relations = []
		obj_to_name = {}

		for obj in r['objects']:
			obj_id = obj['object_id']
			obj_to_name[obj_id] = obj['name']
			r_objects.append(obj['name'])
			if obj_id in obj_to_attr:
				r_attributes.append(
					(obj['name'], obj_to_attr[obj_id])
				)

		for rel in r['relationships']:
			subject_id = rel['subject_id']
			object_id  = rel['object_id']
			predicate  = rel['predicate']
			r_relations.append(
				(obj_to_name[subject_id], predicate, obj_to_name[object_id])
			)
			
		image_graphs.append(
			(index, Graph(r['phrase'], r_objects, r_attributes, r_relations))
		)

	return image_graphs

def dump_region_graphs(region_graphs, filename):
	with open(filename, 'wb') as f:
		pickle.dump(region_graphs, f)

if __name__ == '__main__':
	region_graphs = []
	# 108078 = len(image_data.json)
	total_to_process = 108078
	pbar = tqdm(range(total_to_process), total=total_to_process)

	process_vg_data()
	region_graphs.sort(key=lambda x: x[0])
	region_graphs = [r for i, r in region_graphs]
	dump_region_graphs(region_graphs, 'data/region_graphs.p')