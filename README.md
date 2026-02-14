**Problem Statement -**



Predict whether a wine is Good or Bad using physicochemical attributes.



**Dataset Description -**



Wine Quality Dataset from UCI containing:



6,497 samples



11 chemical features



Target variable: Quality Score



**Models Used -**



Model Name 	Accuracy 	AUC 	Precision 	Recall 	F1 	MCC 

Logistic 

Regression 

Decision Tree       

kNN       

Naive Bayes       

Random Forest(Ensemble) 

XGBoost(Ensemble) 





index,	ML Model Name,			Accuracy,		AUC,			Precision,		Recall,			F1,			MCC

0,	Logistic Regression,		0.8246153846153846,	0.8121440688234582,	0.6071428571428571,	0.2698412698412698,	0.37362637362637363,	0.3210148866422379

1,	Decision Tree,			0.84,			0.7591027505149642,	0.5808823529411765,	0.626984126984127,	0.6030534351145038,	0.5036187854322501

2,	kNN,				0.8323076923076923,	0.8226118532654793,	0.5876288659793815,	0.4523809523809524,	0.5112107623318386,	0.41719816338114735

3,	Naive Bayes,			0.7607692307692308,	0.7756118987035017,	0.42297650130548303,	0.6428571428571429,	0.510236220472441,	0.3745931000081237

4,	Random Forest,			0.8876923076923077,	0.9262465164182722,	0.8045977011494253,	0.5555555555555556,	0.6572769953051644,	0.6073407835015369

5,	XGBoost,			0.8892307692307693,	0.9112178904640736,	0.7571428571428571,	0.6309523809523809,	0.6883116883116883,	0.6254549709343773

&nbsp;    





**Observations on the performance of each model on the chosen** 



ML Model Name			Observation about model performance

Logistic Regression		Works well but limited by linear assumption.

Decision Tree			Captures non-linear patterns but slightly overfits.

kNN				Sensitive to scaling and dataset size.

Naive Bayes			Fast but assumes feature independence.

Random Forest			Strong accuracy, reduces overfitting via ensemble learning.

XGBoost				Best overall performance due to boosting and optimization.





