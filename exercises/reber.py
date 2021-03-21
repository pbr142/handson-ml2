from collections import defaultdict
from random import choice, sample
from typing import DefaultDict, Iterable, List, Tuple

def dict_from_edges(edges: Tuple[Tuple]) -> DefaultDict:
	"""Turn tuple of edges into dictionary, keyed by starting node.

	The dictionary representation is convenient for simulating a path
	through the graph of the grammar.

	Args:
		edges (Tuple[Tuple]): grammar definition as tuple of tuples.
		Each edge is a tuple of the form (start_node, end_node, letter)

	Returns:
		DefaultDict: Dictionary of the edges. Each starting node is a key
		with a list as value. The list contains tuples of the form
		(end_key, letter)
	"""

	node_dict = defaultdict(list)
	for (start_node, end_node, letter) in edges:
		node_dict[start_node].append((end_node, letter))
	return node_dict


def generate_sentence(node_dict: dict) -> Tuple[Tuple]:
	"""Generate a sentence from a dictionary representation of a grammar.

	Args:
		node_dict (dict): Dictionary representation of a grammar.
		Keys are the starting node. Values are (end_node, letter)

	Returns:
		Tuple[Tuple]: Generated sentence as a tuple of edge tuples.
		Each edge has the format (start_node, end_node, letter)
	"""
	sentence = []
	start_node = 0
	while start_node is not None:
		end_node, letter = choice(node_dict[start_node])
		sentence.append((start_node, end_node, letter))
		start_node = end_node
	return tuple(sentence)


def string_from_sentence(sentence: Tuple[Tuple]) -> str:
	"""Get sentence as string from list of list representation

	Args:
		sentence (Tuple[Tuple]): Sentence in list of list representation

	Returns:
		str: Sentence as string
	"""
	return ''.join([edge[-1] for edge in sentence])


def unique_letters(edges: Iterable) -> str:
	"""Return unique letters in collection of edges

	Args:
		edges (Iterable): Iterable containing edges, i.e. tuples.
		The last element in the tuples is assumed to contain the letter.

	Returns:
		str: String of unique letters
	"""
	return ''.join(set([edge[-1] for edge in edges]))


def corrupt_edge(sentence_edge: Tuple, edges: Tuple[Tuple]) -> Tuple:
	"""Corrupt an edge. Based on all possible edges, the function first
	determines which letters would have been possible for the given edge.
	Then, it replaces the letter with a non-allowed letter

	Args:
		sentence_edge (Tuple): Edge to be corrupted
		edges (Tuple[Tuple]): Grammar in tuple of tuples representation

	Returns:
		Tuple: Corrupted edge
	"""
	start_node = sentence_edge[0]
	possible_nodes = [edge for edge in edges if edge[0] == start_node]
	possible_letters = [node[-1] for node in possible_nodes]
	all_letters = unique_letters(edges)
	corrupted_letter = choice(list(set(all_letters) - set(possible_letters)))
	return (start_node, sentence_edge[1], corrupted_letter)


def corrupt_sentence(sentence: Tuple[Tuple], edges: Tuple[Tuple], n_corruptions: int) -> Tuple[Tuple]:
	"""Corrupt a valid sentence with a given number of changes.

	Args:
		sentence (Tuple[Tuple]): Input sentence
		edges (Tuple[Tuple]): The grammar, in tuple of edges representation
		n_corruptions (int): Number of corruptions

	Returns:
		Tuple[Tuple]: Corrupted sentence, in tuple of edges representation
	"""
	assert n_corruptions <= len(sentence)
	index_corruptions = sample(range(len(sentence)), n_corruptions)
	corrupted_sentence = list(sentence)
	for index in index_corruptions:
		corrupted_sentence[index] = corrupt_edge(corrupted_sentence[index], edges)
	return tuple(corrupted_sentence)


def flatten_embedded_edges(edges: Tuple[Tuple]) -> Tuple[Tuple]:
	"""Function to flatten a Embedded Reber Grammer where the Tuple of Edges representation could contain nested
	structures, i.e. an element of the Tuple is itself again a Tuple of Tuples

	Args:
		edges (Tuple[Tuple]): Reber grammer in possibly nested Tuple of edges representation

    Returns:
		Tuple[Tuple]: Flattened Tuple of edges
	"""
	new_edges = []
	for start_node, end_node, letter in edges:
		if isinstance(letter, tuple):
			t = [[str(start_node) + '-' + str(s), str(start_node)+'-'+str(e), l] for s, e, l in letter]
			t[0][0] = start_node
			t[-1][1] = end_node
			for tt in t:
				new_edges.append(tt)
		else:
			new_edges.append([start_node, end_node, letter])
	new_edges = [(s,e,l) for s,e,l in new_edges]
	return tuple(new_edges)


def string_to_ids(s: str, allowed_chars: str) -> List[int]:
	"""Translate string into list of indices, starting at 1, as defined in allows_chars.

	Args:
		s (str): Input string to Translate
		allowed_chars (str): Allowed characters in the order that the output should be returned

	Returns:
		List[int]: Integer representation of the input string, starting at 1 for the first letter in allowed_chars, and so forth
	"""

	return [allowed_chars.index(c)+1 for c in s]