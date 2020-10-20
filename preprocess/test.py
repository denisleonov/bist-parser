import pickle
from visual_genome_parsing import Graph

if __name__ == "__main__":
    with open('data/region_graphs.p', 'rb') as f:
        region_graphs = pickle.load(f)
    
    for i, rg in enumerate(region_graphs):
        if i > 6: 
            break
        
        print(f'Graph #{i}')
        print('phrase:', rg.phrase)
        print('objects:', rg.objects)
        print('attributes:', rg.attributes)
        print('relationships:', rg.relationships)

        print()