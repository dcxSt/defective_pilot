# defective_pilot

## Outline of the problem
After a tragic incident where a pilot commited suicide on the job, we want to detect psychologically unstable / defective pilots. The data-set we are working with is as follows: pilots need training every 6 months. To do this they do a flight simulation and data is gathered on them during the test flight (acceleration, velocity, etc.), we are not told exactly what the data is that this data is collected at a rate of 10Hz; a supervisor watches the pilot and writes 0 if the pilot is doing fine, writes 1 if the pilot is stressed or worried or something.

We want to use the data to see if the pilot is defective (aka, label = 1) 


## Ideas and Brainstorming
Cluster the 1's, centered at some point --- centroid, make balls around until it covers 90% of data, closer to centroid of 0's should be lesser weight of label on the supervised learning
then use RNN ? 2 enboxed lstm's.


## Preprocessing
Getting rid of useless data. Some of the test runs in the CSV files were quite short so we decided to drop them, we dropped all the runs with under 600 data-points, so all of those under a minuit. This amounts to roughtly 13% of the data.

We wanted all of our test-runs to be the same size so we chopped them into 1 minuet lengths. Also we normalised the data.

## Original features

![](figures/index_125.png)
![](figures/index_397.png)


## Our approach
### Weeding out the bad data
#### Extracting some features
Mean, Standard deviation, Max, Min, double derivative mean, number of non-zero gradients 3D only for 4-5-6, mean gradient when grad is non-zero - 2D only for 4 & 5, How many times the gradient changes sign

### ML


## Solution
We will cut up the data (down-sample).

