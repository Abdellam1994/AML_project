# -*- coding: utf-8 -*-
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt

from constants import deck

# Number of observations
N = 10000
n = 208
# Number of clusters
K = 20
# nb of components for PCA
nb_components = 20

np.random.shuffle(deck)

X = np.concatenate((np.random.randint(0,14,size = 13),np.random.random(n))).reshape((1, n + 13))

for i in xrange(N):
	X = np.concatenate((X, np.concatenate((np.random.randint(0,14,size = 13),np.random.random(n))).reshape((1, n + 13))), 0)


pca = PCA(n_components = nb_components)
X = pca.fit_transform(X)

# Scale features
X -= np.mean(X, axis=0)
X /= np.std(X, axis=0)

# Doing the Kmeans clustering	
y = KMeans(n_clusters = K).fit(X).labels_
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# fitting the parameters for the svm classification

C_range = [10**i for i in xrange(-3,3)] 
gamma_range = [10**i for i in xrange(-3,3)]
classifiers = []
scores = []
for C in C_range:
    print(C)
    for gamma in gamma_range:
        clf = svm.SVC(C=C, gamma=gamma)
        scores.append(cross_val_score(clf, X_train, y_train, cv=5).mean())

# plot the scores of the grid
# We extract just the scores
        
scores = np.array(scores).reshape(len(C_range), len(gamma_range))
plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot)
plt.xlabel('gamma')
plt.ylabel('C')
plt.colorbar()
plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
plt.yticks(np.arange(len(C_range)), C_range)
plt.title('Validation accuracy')
plt.show()

# Retrieving the best indices
C_index, gamma_index = np.unravel_index(scores.argmax(), scores.shape)
C = C_range[C_index]
gamma = gamma_range[C_index]

##################################### MONTE CARLO TREE SEARCH ALGORITHM ######################################

# This experiment was not relevant
"""
class Node():
	def __init__(self, state, parent=None):
		self.visits=1
		self.reward=0.0	
		self.state= state
		self.children= []
		self.parent = parent	
		
	def add_child(self,child_state):
		child = Node(child_state, self)
		self.children.append(child)
		
	def update(self,reward):
		self.reward += reward
		self.visits += 1
		
	def fully_expanded(self) :
		
		if len(self.children) == self.state.num_moves :
			return True
		return False
		
	def __repr__(self):
		
		s = "Node; children: %d; visits: %d; reward: %f"%(len(self.children),self.visits,self.reward)
		return s
		


def UCTSEARCH(budget,root):
	for iter in range(budget):
		if iter%10000==9999:
			logger.info("simulation: %d"%iter)
			logger.info(root)
		front=TREEPOLICY(root)
		reward=DEFAULTPOLICY(front.state)
		BACKUP(front,reward)
	return BESTCHILD(root,0)

def TREEPOLICY(node):
	while node.state.terminal()==False:
		if node.fully_expanded()==False:	
			return EXPAND(node)
		else:
			node=BESTCHILD(node,SCALAR)
	return node

def EXPAND(node):
	tried_children=[c.state for c in node.children]
	new_state=node.state.next_state()
	while new_state in tried_children:
		new_state=node.state.next_state()
	node.add_child(new_state)
	return node.children[-1]

#current this uses the most vanilla MCTS formula it is worth experimenting with THRESHOLD ASCENT (TAGS)
def BESTCHILD(node,scalar):
	bestscore=0.0
	bestchildren=[]
	for c in node.children:
		exploit=c.reward/c.visits
		explore=math.sqrt(math.log(2*node.visits)/float(c.visits))	
		score=exploit+scalar*explore
		if score==bestscore:
			bestchildren.append(c)
		if score>bestscore:
			bestchildren=[c]
			bestscore=score
	if len(bestchildren)==0:
		logger.warn("OOPS: no best child found, probably fatal")
	return random.choice(bestchildren)

def DEFAULTPOLICY(state):
	while state.terminal()==False:
		state=state.next_state()
	return state.reward()

def BACKUP(node,reward):
	while node!=None:
		node.visits+=1
		node.reward+=reward
		node=node.parent
	return
"""