import reber

reber_edges = ((0,1,'B'), (1,2,'T'), (1,3,'P'), (2,2,'S'), (2,4,'X'), (3,3,'T'),
    (3,5,'V'), (4,3,'X'), (4,6,'S'), (5,4,'P'), (5,6,'V'), (6,None,'E'))

def test_dict_from_edges():
    node_dict = reber.dict_from_edges(reber_edges)
    assert len(node_dict) == 7
    assert list(node_dict.keys()) == list(range(7))

def test_generate_sentence():
    node_dict = reber.dict_from_edges(reber_edges)
    sentence = reber.generate_sentence(node_dict)
    assert reber_edges[0] == (0,1,'B')
    assert sentence[0] == reber_edges[0]
    assert sentence[1] in reber_edges[1:3]
    assert sentence[-1] == reber_edges[-1]