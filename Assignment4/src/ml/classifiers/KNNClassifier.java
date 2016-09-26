package ml.classifiers;

import java.util.ArrayList;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.*;
import ml.data.DataSet;
import ml.data.Example;
import ml.utils.*;

/**
 * k-Nearest Neighbor perceptron classifier
 * 
 * @author Maddie Gordon, Nick Reminder
 */
public class KNNClassifier implements Classifier {

	private int k = 3; 
	private PriorityQueue<ExampleDist> pq;
	private ArrayList<Example> myExamples;


	@Override
	public void train(DataSet data) {
		myExamples = (ArrayList<Example>) data.getData().clone();
	}


	/**
	 * calculate the Euclidean distance between two examples
	 * 
	 * @param a	- first example
	 * @param b - second example
	 * @return	Double representing Euclidean distance between the two
	 */
	protected double getEucDistance(Example a, Example b) {
		Object[] aSet = a.getFeatureSet().toArray();
		Object[] bSet = b.getFeatureSet().toArray();
		int minLength = Math.min(aSet.length, bSet.length);
		double acc = 0.0;
		for (int i = 0; i < minLength; i++) {
			acc += Math.pow(a.getFeature((int) aSet[i]) - b.getFeature((int) bSet[i]), 2);
		}
		return Math.sqrt(acc);
	}

	@Override
	public double classify(Example example) {
		PriorityQueue<ExampleDist> pq = new PriorityQueue<ExampleDist>();
		for (Example ex : myExamples) {
				double dist = getEucDistance(example, ex);
				pq.add(new ExampleDist(ex, dist));
		}		
		return majLabel(pq);
	}
	
	/**
	 * Calculate the majority label contained in the priority queue
	 * @param p - priority queue containing the k nearest neighbors of the example to be classified
	 * @return label which will classify the example
	 */
	protected double majLabel(PriorityQueue<ExampleDist> p) {
		int sum = 0;
		for(int i = 0; i < k; i++) {
			sum += p.poll().getExample().getLabel();	
		}
		return (sum > 0) ? 1 : -1;
	}


	/**
	 * set the value of k (e.g. change number of nearest neighbors to look for)
	 * 
	 * @param k
	 */
	public void setK(int k) {
		this.k = k;
	}

	public static void main(String[] args) {
		KNNClassifier kc = new KNNClassifier();
		String csvFile = "/Users/maddie/Documents/FALL2016/MachineLearning/hw4/titanic-train.real.csv";
		DataSet data = new DataSet(csvFile);
		kc.train(data);
		for(Example ex : data.getData()){
			System.out.println(kc.classify(ex));
			//kc.classify(ex);
		}
	}

}
