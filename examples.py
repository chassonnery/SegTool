import pandas as pd
from WatershedClustering import ClusterizeSphereObjects
import matplotlib.pyplot as plt

Results = ClusterizeSphereObjects("demo_data/spheres.csv", header="cluster (test 1)")

Results = ClusterizeSphereObjects(Results, dil_coeff=1.2, header="cluster (test 2)")

Results = ClusterizeSphereObjects(Results, InputRods="demo_data/rods.csv", dil_coeff=1.2, header="cluster (test 3)")

Results = ClusterizeSphereObjects(Results, InputRods="demo_data/rods.csv", dil_coeff=1.2, PeriodicBoundaryCondition=True, xmax=15, ymax=15, zmax=15, dil_coeff=1.2, header="cluster (test 4)")

plt.figure(1)
plt.subplot(1,4,1)
plt.plot3()
plt.title("Clusterization with default parameters")

plt.subplot(1,4,2)
plt.plot3()
plt.title("Clusterization with dilation by coefficient 1.2")

plt.subplot(1,4,3)
plt.plot3()
plt.title("Clusterization with dilation by coefficient 1.2 \n and using rod objects as separators")

plt.subplot(1,4,4)
plt.plot3()
plt.title("Clusterization with dilation by coefficient 1.2, \n using rod objects as separators \n and with periodic boundary conditions enabled")














import pandas as pd
from WatershedClustering import ClusterizeSphereObjects

Results = ClusterizeSphereObjects("demo_data/spheres.csv", InputRods="demo_data/rods.csv", dil_coeff=1.2)

display(Results)

Results.to_csv("demo_data/clusterization_test1.csv")


