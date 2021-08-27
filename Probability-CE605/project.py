import math
from scipy.integrate import quad
from matplotlib import pyplot as plt 
import numpy as np

f = open("20103082.txt", "r")
f_out = open("20103082_output","w+")


def estimate_parameters(pdf , data):
    
    n = len(data)
    if pdf == "Normal":
	# by method of Maximum Likelihood 
	# calculate estimated paramters
	sum_x = 0
	for i in range(0,n):
	    sum_x += data[i]
	estimated_mean = (1.0 * sum_x/n)

	squared_sum = 0
        for i in range(0,n):
	    squared_sum += math.pow((data[i] - estimated_mean),2)

	estimated_variance = (1.0*squared_sum/(n-1))
	estimated_stddev = math.sqrt(estimated_variance)
	return [estimated_mean,estimated_stddev]

    if pdf == "Chi-Square":
	#Dof = k - 1 - m
	# k = 2 * sqrt(n) . To get the no. of bins
	# m = 1 for nu(Parameter for Chi-Square.

	k = 2 * math.sqrt(n)
	return k - 1 - 1

    if pdf == "Exponential":
	#By using method of Max. Likelihood
        lamda = 1.0 * n / sum(data)		
	return lamda



def confidence_interval(distribution, data, *params):
    #we have to calculate for 90% CI.
    
    n = len(data)
    if distribution == "Normal":
	#Population variance is not known	
	#1. Calculate sample mean
	#2. Calculate S^2
	
	X_bar = 1.0 * sum(data)/n
	squared_sum = 0
	for i in range(0,n):
	    squared_sum += math.pow((data[i] - X_bar),2)
	S_square = 1.0 * squared_sum/(n-1)
	
	sample_stddev = math.sqrt(S_square)

	##Calculate the CI using formula [ X_bar - t(n-1,alpha/2) * S/sqrt(n) , X_bar - t(n-1,alpha/2) * S/sqrt(n) ]
	## Value of t(100,5%) from T-distribution table = 1.66
	low_mean = X_bar - 1.66 * sample_stddev/math.sqrt(n)
	high_mean = X_bar + 1.66 * sample_stddev/math.sqrt(n)
	
	## CI for Variance
	## Population mean is not known
	## 1.We get the CI using formula [ (n-1) * S^2 / Cu  , (n-1) * S^2 / Cl ]
	## For the Interval Estimation we use the Chi square dist. table.
	## Cu = 123 and Cl = 76 for dof = 99 and alpha/2 = 0.05 and 0.95 respectively.
	
	Cu = 123
	Cl = 76
	low_stddev = math.sqrt((n - 1) * S_square/Cu)
	high_stddev = math.sqrt((n - 1) * S_square/Cl)
	return [[low_mean,high_mean],[low_stddev,high_stddev]]

    if distribution == "Chi-Square":
	## 2*dof = std. deviation
	## Estimating the CI for std deviation and from that we can estimate CI for dof.
        ## CI for VARIANCE = [(n-1)*S^2/Chi-Square(alpha/2) ,(n-1)*S^2/Chi-Square(alpha/2) ]	
        ## alpha/2 = 0.05
	
	dof = params[0]
	sample_mean = sum(data)/n
	print "Value of Chi-Square dist for dof = {} for alpha = {} and {} \n is {} {} respectively".format(dof,0.05,0.95, 27.587, 8.672)
        squared_sum = 0	
        for i in range(0,n):
            squared_sum += math.pow((data[i] - sample_mean),2)
        S_square = 1.0 * squared_sum/(n-1)

	
        low_variance = 1.0 * (n - 1) * S_square / 27.587
	high_variance = 1.0 * (n - 1) * S_square / 8.672
	
	return [low_variance, high_variance]
    
    if distribution == "Exponential":
	## To Calculate the CI for the Exponential Distribution
	## As mean = 1/lamda
	## we use formula [1/X_bar (1 - Z(alpha/2)/sqrt(n)), 1/X_bar (1 + Z(alpha/2)/sqrt(n)) ]
	## Z(alpha/2) = 1.645  for alpha = 90%
	sample_mean = sum(data)/n
	
	low_lambda = 1/sample_mean * (1 - 1.645/ math.sqrt(n)) 
	high_lambda = 1/sample_mean * (1 + 1.645/ math.sqrt(n)) 
	return [low_lambda,high_lambda]


def hypothesis_testing(data, mean = -1 , variance = 0):

    n = len(data)
    #H0 => mu = mu_0 = 2500
    #HA => mu != mu_0
       
    mu_0 = 2500
    #(5% significance level)
    significance_level = 5  	
    
    if variance == 0:
    	#Compute Test Statistic using T-Distribution
    	#Population Variance is not known
    
    	sample_mean = sum(data)/n
	squared_sum = 0
    	for i in range(0,n):
        	squared_sum += math.pow((data[i] - sample_mean),2)
    	S_square = 1.0 * squared_sum/(n-1)
    	sample_stddev = math.sqrt(S_square)
	
	#Test Statistic
	T = 1.0 * (sample_mean - mu_0)/(1.0 * sample_stddev/math.sqrt(n)) 
        
	print "Test Statistic for Hypothesis Testing that mean discharge = 2500 is {}".format(T)
	f_out.write("Test Statistic for Hypothesis Testing that mean discharge = 2500 is {}".format(T))
        #For significance level 5 we have to check between 
	# range -1.987 to 1.987 for dof = n - 1 = 99
	
	if T < 1.987 and T > -1.987:
	    return ["H0 cannot be rejected",1]
	else:
            return ["H0 can be rejected",0]		




data1 = []
data2 = []
data3 = []
data4 = []




header = f.readline()

while True:
    line = f.readline()
    if len(line) != 0:

        val = line.split()
	data1.append(float(val[0]))
	data2.append(float(val[1]))	
	data3.append(float(val[2]))	
	data4.append(float(val[3]))	
    else:
        break



print("###########################################################################")
print("######################## For DATA SITE 1 ##################################")
print("###########################################################################")

f_out.write("###########################################################################\n")
f_out.write("######################## For DATA SITE 1 ##################################\n")
f_out.write("###########################################################################\n")


#Assuming the above distribution to be Normal using Histogram
#1. P(X > x_k) = 0.01 ,so P(X < x_k) = 0.99
#2. P(Z < (x_k - mean)/variance) = 0.99
#3a. According to Std. Normal Distribution table
#3b. (x_k - mean)/std_dev = 2.33

#   Estimate the Paramters.
[est_mean,est_stddev] = estimate_parameters("Normal",data1)

print "est_mean = {}".format(est_mean)
print "est_stddev = {}".format(est_stddev)

f_out.write("est_mean = {}\n".format(est_mean))
f_out.write("est_stddev = {}\n".format(est_stddev))

#Rearranging the eqn (x_k - mean)/stddev = 2.33 from #3b 
# we get x_k = 2.33 * std_dev + mean

x_k = 2.33 * est_stddev + est_mean
print "Value of x_k = {}".format(x_k)

f_out.write("Value of x_k = {}\n".format(x_k))

[mean_interval, variance_interval] = confidence_interval("Normal", data1)
print "mean_interval = {}".format(mean_interval)
print "variance_interval = {}".format(variance_interval)

f_out.write("90 % CI for mean = {}\n".format(mean_interval))
f_out.write("90 % CI for variance = {}\n".format(variance_interval))

H0 = "mu = mu_0 = 2500"
HA = "mu != mu_0"

res = hypothesis_testing(data1)
if res[1] == 1:
    f_out.write("{} {}\n".format(H0,res[0]))
else:
    f_out.write("{} {}\n".format(HA,res[0]))


f_out.write('\n')
print("###########################################################################")
print("######################## For DATA SITE 2 ##################################")
print("###########################################################################")
f_out.write("###########################################################################\n")
f_out.write("######################## For DATA SITE 2 #################################\n")
f_out.write("###########################################################################\n")

#   Estimate the Paramters.
[est_mean,est_stddev] = estimate_parameters("Normal",data2)

print "est_mean = {}".format(est_mean)
print "est_stddev = {}".format(est_stddev)

f_out.write("estimated mean = {}\n".format(est_mean))
f_out.write("estimated standard deviation = {}\n".format(est_stddev))

#Rearranging the eqn (x_k - mean)/variance = 2.33 from #3b
# we get x_k = 2.33 * est_stddev + est_mean

x_k = 2.33 * est_stddev + est_mean
print "Value of x_k = {}".format(x_k)

f_out.write("Value of x_k = {}\n".format(x_k))

[mean_interval, variance_interval] = confidence_interval("Normal", data2)
print "mean_interval = {}".format(mean_interval)
print "variance_interval = {}".format(variance_interval)

f_out.write("90 % CI for mean = {}\n".format(mean_interval))
f_out.write("90 % CI for variance = {}\n".format(variance_interval))

res = hypothesis_testing(data2)
if res[1] == 1:
    f_out.write("{} {}\n".format(H0,res[0]))
else:
    f_out.write("{} {}\n".format(HA,res[0]))

f_out.write('\n')
print("###########################################################################")
print("######################## For DATA SITE 3 ##################################")
print("###########################################################################")
f_out.write("###########################################################################\n")
f_out.write("######################## For DATA SITE3 #################################\n")
f_out.write("###########################################################################\n")


dof = estimate_parameters("Chi-Square",data3)

print "Degree of freedom = {}".format(dof)
f_out.write("Degree of freedom = {}\n".format(dof))

variance_interval = confidence_interval("Chi-Square", data3,dof)

stddev_interval = [math.sqrt(variance_interval[0]), math.sqrt(variance_interval[1])]

print "Confidence Interval for Std dev = {} ".format(stddev_interval)
f_out.write("Confidence Interval for Std dev = {} \n".format(stddev_interval))
f_out.write('\n')

print("###########################################################################")
print("######################## For DATA SITE 4 ##################################")
print("###########################################################################")
f_out.write("###########################################################################\n")
f_out.write("######################## For DATA SITE 4 #################################\n")
f_out.write("###########################################################################\n")


#Estimate the Paramters.
lamda = estimate_parameters("Exponential",data4)


print "lambda = {}\n".format(lamda)

f_out.write("lambda = {}\n".format(lamda))
lambda_interval = confidence_interval("Exponential", data4)

print "Confidence Interval for Lambda = {}".format(lambda_interval)

f_out.write("90 % Confidence Interval for Lambda = {}\n".format(lambda_interval))
######################################
######## Calculate x_k ###############
######################################

## Getting the Value of x_k by integrating the distibution 
## function in interval 0 to x_k and equating with 0.99
## Got the Value of x_k from Matlab Code(Snippet Pasted in Report)

x_k = 11050
print "x_k value obtained by using matlab code(Snipped Pasted) for integration"
print "Value of x_k = {}".format(x_k)

f_out.write("Value of x_k = {}\n".format(x_k))

res = hypothesis_testing(data4)
if res[1] == 1:
    f_out.write("{} {}\n".format(H0,res[0]))
else:
    f_out.write("{} {}\n".format(HA,res[0]))
