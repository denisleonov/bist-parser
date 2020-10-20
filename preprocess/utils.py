from spacy_wordnet.wordnet_annotator import WordnetAnnotator

def word_to_wordnet(word: str, nlp: WordnetAnnotator) -> set[str]:
	token = nlp(word)[0]
	return set([l.name() for l in token._.wordnet.lemmas()])

def similar(pred_syns: set[str], ref_syns: set[str]) -> bool:
	if pred_syns.intersection(ref_syns):
		return True
	else:
		return False
