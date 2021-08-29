# SamWalker & SamWalker++

This is our source codes for the conference paper: <br>
[1] Jiawei Chen, Can Wang, Sheng Zhou, Qihao Shi, Yan Feng, and Chun Chen. "SamWalker: Social Recommendation with Informative Sampling Strategy." In The World Wide Web Conference, pp. 228-239. ACM, 2019. <br>
and its extension: <br>
[2] Can Wang, Jiawei Chen, Sheng Zhou, Qihao Shi, Yan Feng, and Chun Chen. "SamWalker++: recommendation with informative sampling strategy." In IEEE Transactions on Knowledge and Data Engineering.

# Example to Run SamWalker++
We implement SamWalker++ in Python 3.6. The required packages are as follows:

- pytorch=1.5.1 <br>
- numpy=1.19.2 <br>
- pandas=0.20.3 <br>
- cppimport == 18.11.8 <br>
- pybind11 == 2.5.0 <br>

We can run the code for the example data: <br>
```shell
python Samwalkerplus.py
```
Where the inputs of the Samwalkerplus function are the paths of the trainning data, the test data and some parameters. <br>

Each line of trainingdata.txt is: UserID \t ItemID \t 1 <br>
Each line of testdata.txt is :UserID \t ItemID \t 1 <br>
Noting: when the number of users or items are above a certain theroshold (500000), you need to enlarge the setting of 'maxnm' in the source code biwalker.cpp (line 12). <br>

# Example to Run SamWalker
We implement SamWalker in MATLAB. Also, we implement sampling process (Personalized random walk) in C++ to improve effeiciency. Before running the code, please compile c++ source codes to generate mex file in matlab enviornment:
```matlab
mex mysamwalknew.cpp
mex myv2s.cpp
```
Then,  we can run the code for the example data:
```matlab
 samwalker('trainingdata.txt','testdata.txt','trustnetwork.txt')
```
Where the inputs of the Samwalker function are the paths of the trainning data, the test data and the social network data respectively.<br>
Each line of trainingdata.txt is: UserID \t ItemID \t 1 <br>
Each line of testdata.txt is :UserID \t ItemID \t 1 <br>
Each line of trustnetwork.txt is: User1ID \t User2ID \t 1 <br>
