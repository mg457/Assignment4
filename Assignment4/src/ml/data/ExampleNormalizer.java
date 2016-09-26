package ml.data;

import java.util.ArrayList;

/**
 * Class which normalizes every example's Euclidean distance 
 * 
 * @author Nick Reminder, Maddie Gordon
 *
 */
public class ExampleNormalizer implements DataPreprocessor {

	/**
	 * Normalizes feature set of each example so each has a Euclidean
	 * distance of one.
	 * 
	 * @param train - DataSet to be processed.
	 */
	@Override
	public void preprocessTrain(DataSet train) {
		ArrayList<Example> myExamples = train.getData();
		for (Example e : myExamples) {
			double myDistance = getDistFromOrigin(e);
			for (int i : e.getFeatureSet()) {
				e.setFeature(i, e.getFeature(i)/myDistance);
			}
		}
	}
	
	/**
	 * Does exact same as train preprocessing.
	 * 
	 * @param test - DataSet to be processed.
	 */
	@Override
	public void preprocessTest(DataSet test) {
		preprocessTrain(test);
	}

	/**
	 * Gets an example's Euclidean distance from the origin.
	 * 
	 * @param aExample - Example to be evaluated.
	 * @return		   - Said example's distance from the origin.
	 */
	public static double getDistFromOrigin(Example aExample) {
		Object[] myFeatures = aExample.getFeatureSet().toArray();
		double acc = 0.0;
		for (int i = 0; i < myFeatures.length; i++) {
			acc += Math.pow((int) myFeatures[i], 2);
		}
		return Math.sqrt(acc);
	}
	
}
