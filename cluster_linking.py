import numpy as np
inspace = np.loadtxt("clusters.txt", delimiter=' ', dtype=float)
latent = np.loadtxt("clusters_latent.txt",delimiter=' ', dtype=float)
print(inspace.shape)
print(latent.shape)
cluster = []
print(len(inspace))
for i in range(8):
    c = []
    for j in range(8):
        #c = np.subtract(inspace[i],latent[j])
        x = np.linalg.norm(inspace[i] - latent[j])
        c.append(x)
    cluster.append(c)
cnp = np.array(cluster)
print(cnp)
print(cnp.shape)
print(np.argmin(cnp, axis=1)+1)


