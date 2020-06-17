# Mesh-Simplification
Mesh Simplification algorithm implemented using Quadric-Based Polygonal Surface Simplification. This is a group project written by Katniss Lee (katleej), Akshay Battu (akshaybattu98) and Arjun Sarup (arjunsarup-1).

## Abstract
The above image was created from our final project — Mesh Simplification. Mesh simplification is an important tool during real-time rendering. Objects in world space that are far away or are not essential features in a scene probably don't need a large amount of detail and can be simplified using mesh simplification, thereby saving memory and rendering time. The high-level overview of our mesh simplification algorithm is that we first compute the cost of collapsing all the edges, and sort them in the order of least to highest cost. The cost of edge contraction is computed using a quadric error system that was specified in Michael Garland's dissertation on Quadric-Based Polygonal Surface Simplification. We then collapse the edge with the least cost at each iteration, and keep on repeating this process for a desired level of simplification. This level can be modified using an in-built factor variable inside our codebase. 

## Technical Approach
Here is the full layout of our algorithm:

1. Compute the quadric error for all vertices. This will help us determine which vertices we will need to eventually remove. The system for computing this quadric error will be described in the following sections.
2. At each iteration, we compute the optimal contraction target v for each valid pair (v1, v2). Define the cost of contracting a pair as the dot product of v and ((Q1 + Q2)v), where Q1 and Q2 are the quadric error matrices of vertex v1 and v2.
3. Calculate the pair that has the lowest cost of edge contraction.
4. Iteratively contract the minimum cost pair, and update costs of all other pairs involving the new contracted vertex.

###### Edge Collapse and Contraction
The first step was writing edge contraction. This was done by calculating the new 
optimal point between the two vertices. There are two ways of computing this optimal point, one which is introduced in Garland’s dissertation (where the optimal v = (-1) * inverse of A * b) and the other is our own customized solution. Our customized solution computes the midpoint of v1 and v2 while considering the difference in the degree of each vertex. The difference between the two approaches will be clearly seen in the demo below. Our optimal vertex equation is the following:

(v1->degree() * v1->position + v2->degree() * v2->position) / (v1->degree() + v2->degree())

![Image Description](https://github.com/katleej/Mesh-Simplification/blob/master/docs/images/image1.png)

We then create a new vertex V with this optimal position, and reassign the necessary neighbors’s attributes to the new optimal vertex. One of the challenges we faced during this step was making the decision of using the old vertex versus creating a completely new one. After much debugging, we decided that creating a new vertex and reassigning the necessary parts was rather simpler than modifying the position of one of the vertices and reassigning all the parts connected to it. 

###### Error Quadrics
Garland proposed a computationally inexpensive solution to calculate the cost of each edge
contraction. He associated a set of planes with every vertex of the model. More information on how we computed can be found [here](https://arjunsarup1998.github.io/cs184-final-project/).


## Results and Analysis
Here is our progressive downsampling of teapot.dae using Garland's optimal v_bar to give an overview of how mesh simplification changes the mesh at each step. At each step, 90% of the faces at the previous steps remain.

![Image Description](https://github.com/katleej/Mesh-Simplification/blob/master/docs/images/image11.gif)

Here is a clear difference between the paper’s original optimal point versus our optimal point. The picture below on the left is when we use Garland’s method, and the other is when we use our own midpoint vertex point that weights two vertices based on their degrees. 

![Image Description](https://github.com/katleej/Mesh-Simplification/blob/master/docs/images/image5.gif)
![Image Description](https://github.com/katleej/Mesh-Simplification/blob/master/docs/images/image9.gif)

We have also experimented changing the multiplier value, which identifies the number of remaining faces that should be left after each call of downsample, and this number is computed by num_faces * multiplier. Therefore, higher the multiplier value, greater the level of simplification. Examples are shown below:

![Image Description](https://github.com/katleej/Mesh-Simplification/blob/master/docs/images/image3.gif)
![Image Description](https://github.com/katleej/Mesh-Simplification/blob/master/docs/images/image4.gif)
![Image Description](https://github.com/katleej/Mesh-Simplification/blob/master/docs/images/image6.gif)
![Image Description](https://github.com/katleej/Mesh-Simplification/blob/master/docs/images/image2.gif)

cow.dae (optimal above, L to R: multiplier = 0.5, 0.25, 0.05, Garland’s optimal multiplier = 0.25)

![Image Description](https://github.com/katleej/Mesh-Simplification/blob/master/docs/images/image10.gif)
![Image Description](https://github.com/katleej/Mesh-Simplification/blob/master/docs/images/image8.gif)










