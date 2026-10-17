import numpy as np

def hamming_distance_kanerva(state: list, prototypes: list, threshold: int) -> tuple:
	"""
	Compute Hamming distances and find active prototypes for Kanerva coding.

	Args:
		state: Binary state vector (list of 0s and 1s).
		prototypes: List of binary prototype vectors.
		threshold: Maximum Hamming distance for a prototype to be active.

	Returns:
		Tuple of (distances, active_indices).
	"""
	def hamming(v1, v2):
		d = 0
		for i in range(len(v1)):
			if v1[i] != v2[i]:
				d+=1
		return d

	ham_dist = []
	active_idx = []   

	for pt in range(len(prototypes)): 

		current_ham = hamming(state, prototypes[pt])
		ham_dist.append(current_ham)

		if current_ham <= threshold:
			active_idx.append(pt)

	return (ham_dist, active_idx)  