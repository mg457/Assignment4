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
		
		//Q3 & Q4
		//KNNClassifier avg = new KNNClassifier();
		AveragePerceptronClassifier avg = new AveragePerceptronClassifier();
		String csvFile = "/Users/maddie/Documents/FALL2016/MachineLearning/hw4/titanic-train.real.csv";
		DataSet data = new DataSet(csvFile);
		CrossValidationSet cvs = new CrossValidationSet(data, 10, true);
		FeatureNormalizer fn = new FeatureNormalizer();
		ExampleNormalizer en = new ExampleNormalizer();
		for (int i = 0; i < 10; i++) {
			DataSetSplit ds = cvs.getValidationSet(i, false);
			//DataSet train = ds.getTrain();
			//DataSet test = ds.getTest();
			
			fn.preprocessTrain(ds.getTrain());
			fn.preprocessTest(ds.getTest());
			en.preprocessTrain(ds.getTrain());
			en.preprocessTest(ds.getTest());
			double correct = 0.0;
			double totalAcc = 0.0;
			for (int j = 0; j < 100; j++) {
				correct = 0;
				avg.train(ds.getTrain());
				for (Example ex : ds.getTest().getData()) {
					if (avg.classify(ex) == ex.getLabel()) {
						correct++;
					}
				}
				totalAcc += (double)correct/(double)ds.getTest().getData().size();
			}
			totalAcc = totalAcc / (double)100;
			System.out.println(totalAcc);
			//System.out.println("Split " + i + ": " + totalAcc);
		}
		
		
//		//KNNClassifier avg = new KNNClassifier();
//		String csvFile1 = "/Users/maddie/Documents/FALL2016/MachineLearning/hw4/titanic-train.real.csv";
//		DataSet data2 = new DataSet(csvFile1);
//		CrossValidationSet cvs2 = new CrossValidationSet(data2, 10);
//		for (int i = 0; i < 10; i++) {
//			DataSetSplit ds = cvs2.getValidationSet(i, true);
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
