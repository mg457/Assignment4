package ml.utils;

import ml.classifiers.*;
import ml.data.*;

public class Experiments {

	public static void main(String[] args) {
		
		//Q1
//		AveragePerceptronClassifier avg = new AveragePerceptronClassifier();
//		String csvFile = "/Users/maddie/Documents/FALL2016/MachineLearning/hw4/titanic-train.perc.csv";
//		DataSet data = new DataSet(csvFile);
//		CrossValidationSet cvs = new CrossValidationSet(data, 10);
//		for (int i = 0; i < 10; i++) {
//			DataSetSplit ds = cvs.getValidationSet(i, true);
//			DataSet train = ds.getTrain();
//			DataSet test = ds.getTest();
//			double correct = 0.0;
//			double eval = 0.0;
//			double totalSum = 0.0;
//			for (int j = 0; j < 100; j++) {
//				correct = 0;
//				avg.train(train);
//				for (Example ex : test.getData()) {
//					if (avg.classify(ex) == ex.getLabel()) {
//						correct++;
//					}
//					eval++;
//				}
//				totalSum += (double)correct/(double)test.getData().size();
//			}
//			totalSum = totalSum / (double)100;
//			System.out.println("Split " + i + ": " + totalSum);
//		}

		//Q2
//		AveragePerceptronClassifier avg = new AveragePerceptronClassifier();
//		String csvFile = "/Users/maddie/Documents/FALL2016/MachineLearning/hw4/titanic-train.real.csv";
//		DataSet data = new DataSet(csvFile);
//		CrossValidationSet cvs = new CrossValidationSet(data, 10);
//		for (int i = 0; i < 10; i++) {
//			DataSetSplit ds = cvs.getValidationSet(i, true);
//			DataSet train = ds.getTrain();
//			DataSet test = ds.getTest();
//			double correct = 0.0;
//			double eval = 0.0;
//			double totalSum = 0.0;
//			for (int j = 0; j < 100; j++) {
//				correct = 0;
//				avg.train(train);
//				for (Example ex : test.getData()) {
//					if (avg.classify(ex) == ex.getLabel()) {
//						correct++;
//					}
//					eval++;
//				}
//				totalSum += (double)correct/(double)test.getData().size();
//			}
//			totalSum = totalSum / (double)100;
//			System.out.println("Split " + i + ": " + totalSum);
//		}
		
		//Q3
		KNNClassifier avg = new KNNClassifier();
		String csvFile = "/Users/maddie/Documents/FALL2016/MachineLearning/hw4/titanic-train.perc.csv";
		DataSet data = new DataSet(csvFile);
		CrossValidationSet cvs = new CrossValidationSet(data, 10);
		for (int i = 0; i < 10; i++) {
			DataSetSplit ds = cvs.getValidationSet(i, true);
			DataSet train = ds.getTrain();
			DataSet test = ds.getTest();
			double correct = 0.0;
			double eval = 0.0;
			double totalSum = 0.0;
			for (int j = 0; j < 100; j++) {
				correct = 0;
				avg.train(train);
				for (Example ex : test.getData()) {
					if (avg.classify(ex) == ex.getLabel()) {
						correct++;
					}
					eval++;
				}
				totalSum += (double)correct/(double)test.getData().size();
			}
			totalSum = totalSum / (double)100;
			System.out.println("Split " + i + ": " + totalSum);
		}
		
		
//		KNNClassifier avg = new KNNClassifier();
//		String csvFile = "/Users/maddie/Documents/FALL2016/MachineLearning/hw4/titanic-train.real.csv";
//		DataSet data = new DataSet(csvFile);
//		CrossValidationSet cvs = new CrossValidationSet(data, 10);
//		for (int i = 0; i < 10; i++) {
//			DataSetSplit ds = cvs.getValidationSet(i, true);
//			DataSet train = ds.getTrain();
//			DataSet test = ds.getTest();
//			double correct = 0.0;
//			double eval = 0.0;
//			double totalSum = 0.0;
//			for (int j = 0; j < 100; j++) {
//				correct = 0;
//				avg.train(train);
//				for (Example ex : test.getData()) {
//					if (avg.classify(ex) == ex.getLabel()) {
//						correct++;
//					}
//					eval++;
//				}
//				totalSum += (double)correct/(double)test.getData().size();
//			}
//			totalSum = totalSum / (double)100;
//			System.out.println("Split " + i + ": " + totalSum);
//		}


	}

}
