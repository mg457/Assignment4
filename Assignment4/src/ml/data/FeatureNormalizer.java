package ml.data;

import java.util.ArrayList;

/**
 * Normalizes features based on centering means to 0 and standard deviations to 1.
 * 
 * @author Nick Reminder, Maddie Gordon
 *
 */
public class FeatureNormalizer implements DataPreprocessor {
	private double[] theFeatureAverages;
	private double[] theFeatureVariances;
	
	/**
	 * Does data preprocessing on the training set.
	 * 
	 * @param train - DataSet to be processed. 
	 */
	@Override
	public void preprocessTrain(DataSet train) {
		createFeatureAverages(train);
		centerBasedOnAverages(train);
		
		createFeatureVariances(train);
		scaleBasedOnVariances(train);
	}

	/**
	 * Does data preprocessing on the test set.
	 * 
	 * @param test - DataSet to be processed.
	 */
	@Override
	public void preprocessTest(DataSet test) {
		centerBasedOnAverages(test);
		scaleBasedOnVariances(test);
	}
	
	/**
	 * Centers feature values around a mean of 0 by subtracting stored 
	 * means for each value across the data set.
	 * 
	 * @param aDataSet	- DataSet to be centered.
	 */
	public void centerBasedOnAverages(DataSet aDataSet) {
		Object[] myFeatureNums = aDataSet.getAllFeatureIndices().toArray();
		ArrayList<Example> myExamples = aDataSet.getData();
		for (int i=0; i<myFeatureNums.length; i++) {
			for (Example e : myExamples) {
				e.setFeature(i, e.getFeature(i) - theFeatureAverages[i]);
			}
		}
	}
	
	/**
	 * Populates feature averages field to center data.
	 * 
	 * @param train - DataSet to be centered.
	 */
	public void createFeatureAverages(DataSet train) {
		Object[] myFeatureNums = train.getAllFeatureIndices().toArray();
		double[] myFeatureAverages = new double[myFeatureNums.length];
		ArrayList<Example> myExamples = train.getData();
		
		//Calculate averages
		for (int i=0; i<myFeatureNums.length; i++) {
			double total = 0;
			for (Example e : myExamples) {
				total += e.getFeature((int) myFeatureNums[i]);
			}
			myFeatureAverages[i] = total/myExamples.size();
		}
		theFeatureAverages = myFeatureAverages;
	}
	
	/**
	 * Scales feature values so that for each feature across the data set,
	 * standard deviation is normalized to 1.
	 * 
	 * @param aDataSet	- DataSet to be scaled.
	 */
	public void scaleBasedOnVariances(DataSet aDataSet) {
		Object[] myFeatureNums = aDataSet.getAllFeatureIndices().toArray();
		ArrayList<Example> myExamples = aDataSet.getData();
		for (int i=0; i<myFeatureNums.length; i++) {
			for (Example e : myExamples) {
				e.setFeature(i, e.getFeature(i)/theFeatureVariances[i]);
			}
		}
	}
	
	/**
	 * Populates feature variances field to scale data.
	 * 
	 * @param train	- DataSet to be normalized.
	 */
	public void createFeatureVariances(DataSet train) {
		Object[] myFeatureNums = train.getAllFeatureIndices().toArray();
		double[] myFeatureVariances = new double[myFeatureNums.length];
		ArrayList<Example> myExamples = train.getData();
		
		//Calculate variances
		for (int i=0; i<myFeatureNums.length; i++) {
			double totalNumerator = 0;
			for (Example e : myExamples) {
				totalNumerator += Math.pow(e.getFeature(i)-theFeatureAverages[i], 2);
			}
			myFeatureVariances[i] = Math.sqrt(totalNumerator/myExamples.size());
		}
		
		theFeatureVariances = myFeatureVariances;
	}
}
