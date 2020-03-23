import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle


def initialization(documents):
    M = len(documents)
    vocab_dict = dict()
    for d in documents:
        for w in d:
            vocab_dict[w] = True
    vocab_list = [w for w in vocab_dict]
    V = len(vocab_list)  # Count of words
    for i in range(V):
        vocab_dict[vocab_list[i]] = i
    for i in range(M):
        words = [vocab_dict[w] for w in documents[i]]
        words = sorted(words)
        documents[i] = words
    return vocab_dict, vocab_list, documents


def print_topic(K, phi):
    print("Topics: ")
    for k in range(K):
        ind = np.argpartition(phi[k], -10)[-10:]
        ind = ind[np.argsort(phi[k][ind])]
        ind = ind[::-1]
        print("  Theme=%d" % k)
        for i in ind:
            print("    word=%s, prob=%f" % (vocab_list[i], phi[k][i]))


t0 = time.time()

parser = argparse.ArgumentParser(description="Uses gibbs samping to solve LDA model.")
parser.add_argument("--data", type=str, default="data.txt", 
                    help="The position of data file.")
parser.add_argument("--K", type=int, default=3, 
                    help="The number of topics.")
parser.add_argument("--step", type=int, default=1500, 
                    help="Max number of steps.")
args = parser.parse_args()

K = args.K  # Number of Topics

documents = []
with open(args.data) as f:
    for line in f:
        documents.append(line.strip().split(" "))
M = len(documents)
vocab_dict, vocab_list, documents = initialization(documents)
V = len(vocab_list)

# Initialize alpha and beta
alpha = np.array([50/K + 1] * K)
beta = np.array([0.01 + 1] * V)

print("Number of topics: %d" % K)
print("Number of documents: %d" % M)
print("Size of vocab: %d" % V)

# Initialize topics, n_{m,k,.}, n_{.,k,t}
topic = []
N_mk = np.zeros([M, K], dtype=np.int)  # #theme k in document m
N_kt = np.zeros([K, V], dtype=np.int)  # #word t in theme k
for m in range(M):
    Nm = len(documents[m])
    topic.append(np.random.randint(0, K, Nm))
    for n in range(Nm):
        k = topic[m][n]
        t = documents[m][n]
        N_mk[m][k] += 1
        N_kt[k][t] += 1
N_kt_tsum = np.sum(N_kt, axis=1)  # sum_t of N_kt
N_mk_ksum = np.sum(N_mk, axis=1)  # sum_k of n_mk

# Gibbs Sampling
theta = np.zeros([M, K], dtype=np.float)
phi = np.zeros([K, V], dtype=np.float)
log_likelihood = []
step = 0
while step < args.step:
    changed = 0
    for m in range(M):
        Nm = len(documents[m])
        for n in range(Nm):
            t = documents[m][n]
            N_mk[m][topic[m][n]] -= 1
            N_kt[topic[m][n]][t] -= 1
            N_kt_tsum[topic[m][n]] -= 1
            # sample new topic[m][n]
            prob = [(N_kt[k][t] + beta[t] - 1) * (N_mk[m][k] + alpha[k] - 1) / (N_kt_tsum[k] + np.sum(beta) - 1) for k in range(K)]
            prob = np.array(prob)
            if np.sum(prob) == 0.0:
                prob = np.array([1/len(prob)] * len(prob))
            else:
                prob = prob / np.sum(prob)
            choices = [i for i in range(K)]
            k = np.random.choice(choices, p=prob)
            # update topic[m][n]
            if topic[m][n] != k:
                changed += 1
                topic[m][n] = k
            N_mk[m][k] += 1
            N_kt[k][t] += 1
            N_kt_tsum[k] += 1
    # update theta and phi
    for m in range(M):
        for k in range(K):
            theta[m][k] = (N_mk[m][k] + alpha[k]) / (N_mk_ksum[m] + np.sum(alpha))
    for k in range(K):
        for t in range(V):
            phi[k][t] = (N_kt[k][t] + beta[t]) / (N_kt_tsum[k] + np.sum(beta))
    # Calculate log-likelihood
    ll = 0.0
    for m in range(M):
        for t in documents[m]:
            ll += np.log(np.sum([theta[m][k] * phi[k][t] for k in range(K)]))
    log_likelihood.append(ll)
    print("Step %d, log-likelihood=%f" % (step, ll))
    
    if changed == 0:
        break
    print("changed=%d" % changed)
    if step % 50 == 0:
        print_topic(K, phi)

    step += 1

print_topic(K, phi)

t= time.time() - t0
plt.plot(log_likelihood)
plt.savefig("ll_k-%d_step-%d.png" % (K, args.step))
with open("k-%d_step-%d.pickle" % (K, args.step), "wb") as handle:
    a = {"ll": log_likelihood, "time": t}
    pickle.dump(a, handle)
print("Time: %f" % t)
