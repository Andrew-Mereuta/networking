# Modelling and Data analysis in Complex systems report
## Group information
Group 1
| Name | Student number |
|-|-|
| Andrei  Mereuta |  |
| Anna Kalandadze |  |
| Marco Bak | 5066212 |

## Assignments
### A.1
Number of nodes is: 403. \
Number of links is: 9889. \
Average degree is: 48.95. \
Standaard deviation of the degree is: 31.71

### A.2
![Degree distribution of the network](./a_2.png)

A scale free graph has a power law degree distribution and an ER random graph has a posson degree distribution.
The degree distribution of our network is much more like a poisson graph.
This means that an ER random graph is more suitable to model our network.

### A.3
The assortative degree corelation of the graph is -0.0693, which is close, but just below zero. This means that the graph is mainly non-assorsative, with a realy small prefference to disassorsative.
In the real world this means that the nodes have no prefference in having links with similar or dissimilar nodes. 

### A.4
The clustering coefficient C of G is 0.2359

### A.5
Hopcount: 1.9530
Diameter: 4

### A.6
Small world graphs are defined by a high clustering coefficient, and low distances between nodes. 
The clustering coefficient is a value between 0 and 1, where a value close to one suggests a small world graph.
While the average hopcount and diameter are relatively small, which could suggest a small world property, the clustering coefficient is quite low to really suggest this.
So this graph most likely has no small world property.