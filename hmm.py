# This code is based on and referenced to the following papers:
	
# 								  	
# 		[1]	Numerical Stable hidden Markov Model Implementation
# 				Tobias P. Mann
# 				February 21, 2006

# 		[2] A Revealing Introduction to Hidden Markov Models
# 				Mark Stamp - Associate Professor
# 				Department of Computer Science - San Jose State University
# 				April 26, 2012

import numpy as np

# Define extended exponential and natural-log functions
# LOGZERO is minus infinity i.e log(0) = -infinity; using -numpy.inf 

def eexp(x):
	if (np.isinf(x)):
		return 0
	else:
		return np.exp(x)

def eln(x):
	if (x == 0):
		return -np.inf
	if (x > 0):
		return np.log(x)
	else:
		print "negative input error"

def elnsum(eln_x, eln_y):
	if ( (np.isinf(eln_x)) or (np.isinf(eln_y)) ):
		if (np.isinf(eln_x)):
			return eln_y
		else:
			return eln_x
	else:
		if (eln_x > eln_y):
			return eln_x + eln(1 + np.exp(eln_y - eln_x))
		else:
			return eln_y + eln(1 + np.exp(eln_x - eln_y))

def elnproduct(eln_x, eln_y):
	if ( (np.isinf(eln_x)) or (np.isinf(eln_y)) ):
		return -np.inf
	else:
		return eln_x + eln_y

def elnForward(A_matrix, B_matrix, pi_vector, initialState_idx, observationSeq):
	
	N 			= len(A[0])				# dimension of hidden states (A)
	T 			= len(observationSeq) 	# length of observation sequence
	eln_alpha 	= np.zeros((N,T)) 		# create alpha-variable

	for i in range(N):
		eln_alpha[i][0] = elnproduct( eln( pi[initialState_idx] ), eln( B_matrix[i][observationSeq[0]] ) )

	for t in range(1,T):
		for j in range(N):
			log_alpha = -np.inf 		# log(0) = -infinity
			for i in range(N):
				log_alpha 	= elnsum( log_alpha, elnproduct( eln_alpha[i][t-1], \
					 								eln(A_matrix[i][j]) ) )
			eln_alpha[j][t] = elnproduct( log_alpha, eln(B_matrix[j][observationSeq[t]]) )
	
	return eln_alpha

def elnBackward(A_matrix, B_matrix, observationSeq):
	
	N 			= len(A[0])				# dimension of hidden states (A)
	T 			= len(observationSeq) 	# length of observation sequence
	eln_beta 	= np.zeros((N,T)) 		# create beta-variable

	for i in range(N):
		eln_beta[i][T-1] = 0 			# log(1) = 0

	for t in range(T-2,-1,-1):
		for i in range(N):
			log_beta = -np.inf
			for j in range(N):
				log_beta = elnsum( log_beta, elnproduct( eln(A_matrix[ i ][ j ]), \
												elnproduct( eln(B_matrix[ j ][ observationSeq[ t+1 ] ]), \
													eln_beta[ j ][ t+1 ] )))
			eln_beta[i][t] = log_beta

	return eln_beta
	
def elnGamma(eln_alpha, eln_beta):

	N 			= len(eln_alpha)		# dimension of hidden states (A)
	T 			= len(eln_alpha[0]) 	# length of observation sequence
	normalizer  = np.zeros((T))   		# create normalizer vector
	eln_gamma 	= np.zeros((N,T)) 		# create gamma-variable

	for t in range(T):
		normalizer[t] = -np.inf 		# log(0) = -infinity
		for i in range(N):
			eln_gamma[i][t] = elnproduct(eln_alpha[i][t], eln_beta[i][t])
			normalizer[t]   = elnsum(normalizer[t], eln_gamma[i][t])
		for i in range(N):
			eln_gamma[i][t] = elnproduct(eln_gamma[i][t], -normalizer[t]) # This equals calculating eln_gamma[i][t] / normalizer[t] 

	return eln_gamma

def elnXi(A_matrix, B_matrix, observationSeq, eln_alpha, eln_beta, eln_gamma):

	N 			= len(A[0])				# dimension of hidden states (A)
	T 			= len(observationSeq) 	# length of observation sequence
	normalizer  = np.zeros((T))   		# create normalizer vector
	eln_xi 		= np.zeros((N,N,T)) 	# create xi-variable -- OBS! 3-dimensional matrix

	for t in range(T-1):
		normalizer[t] = -np.inf
		for i in range(N):
			for j in range(N):
				eln_xi[i][j][t] = elnproduct( eln_alpha[ i ][ t ], \
									elnproduct( eln(A_matrix[ i ][ j ]), \
										elnproduct( eln(B_matrix[ j ][ observationSeq[t+1] ]), \
											eln_beta[ j ][ t+1 ] )))
				normalizer[t] = elnsum( normalizer[t], eln_xi[i][j][t] )
		for i in range(N):
			for j in range(N):
				eln_xi[i][j][t] = elnproduct( eln_xi[i][j][t], -normalizer[t] ) # This equals calculating eln_xi[i][j][t] / normalizer[t] 

	return eln_xi

def reEstimatePi(eln_gamma):

	N 		= len(eln_gamma) 			# dimension of hidden states
	new_pi  = np.zeros((N))				# create new Pi-vector
	
	for i in range(N):
		new_pi[i] = eexp(eln_gamma[i][0]) 	# OBS! taking extended exponential

	return new_pi

def reEstimateA(eln_gamma, eln_xi):

	N 			= len(eln_gamma)		# dimension of hidden states	
	T 			= len(eln_gamma[0])		# length of observation sequence
	new_A 		= np.zeros((N,N)) 		# create new A-matrix

	for i in range(N):
		for j in range(N):
			numerator 	= -np.inf 				
			denominator = -np.inf 		
			for t in range(T-1):
				numerator 	= elnsum( numerator  , eln_xi[i][j][t] ) 
				denominator = elnsum( denominator, eln_gamma[i][t] ) 
			new_A[i][j] = eexp(elnproduct(numerator, -denominator)) 	# OBS! taking extended exponential

	return new_A

def reEstimateB(eln_gamma, eln_xi, observationSeq, B_matrix):

	N 			= len(eln_gamma) 		# number/dimension of hidden variables
	M 			= len(B_matrix[0]) 		# number of observable variables
	T 			= len(eln_gamma[0])		# length of observation sequence
	new_B 		= np.zeros((N, M)) 		# create new B-matrix

	for i in range(N):
		for j in range(M):
			numerator 	= -np.inf
			denominator = -np.inf			
			for t in range(T):
				if (observationSeq[t] == j):
					numerator = elnsum(numerator, eln_gamma[i][t])
				denominator = elnsum(denominator, eln_gamma[i][t])			
			new_B[i][j] = eexp(elnproduct(numerator, -denominator))		# OBS! taking extended exponential

	return new_B

# Providing example variables that suits: 	1. A simple example - [2] p.1 

A 		= np.array([[0.7, 0.3],[0.4, 0.6]])				# state transition matrix

#					 	  H    C
#					 H [ 0.7  0.3 ]
#					 C [ 0.4  0.6 ]

B 		= np.array([[0.1, 0.4, 0.5],[0.7, 0.2, 0.1]]) 	# observation matrix

#				 	  S    M    L
#				 H [ 0.1  0.4  0.5 ]
#				 C [ 0.7  0.2  0.1 ]

pi 		= np.array([0.6, 0.4])  						# initial state vector

O   	= np.array([0, 1, 0, 2]) 						# observation sequence












