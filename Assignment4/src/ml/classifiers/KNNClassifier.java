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

	public KNNClassifier() {}

	@Override
	public void train(DataSet data) {
		ArrayList<Example> myExamples = (ArrayList<Example>) data.getData().clone();
		pq = initPq(myExamples);
	}

	/**
	 * method to initialize the priority queue representing the examples in the
	 * dataset
	 * 
	 * @param data
	 *            - examples in the dataset
	 * @return priority queue with all weights initialized to 0
	 */
	public PriorityQueue<ExampleDist> initPq(ArrayList<Example> data) {
		PriorityQueue<ExampleDist> pq = new PriorityQueue<ExampleDist>();
		for (Example ex : data) {
			pq.add(new ExampleDist(ex, 0.0));
		}
		return pq;
	}

	/**
	 * calculate the Euclidean distance between two examples
	 * 
	 * @param a
	 * @param b
	 * @return
	 */
	protected double getEucDistance(Example a, Example b) {
		Object[] aSet = a.getFeatureSet().toArray();
		Object[] bSet = b.getFeatureSet().toArray();
		int minLength = Math.min(aSet.length, bSet.length);
		double acc = 0.0;
		for (int i = 0; i < minLength; i++) {
			acc += Math.pow(a.getFeature((int) aSet[i]) - b.getFeature((int) bSet[i]), 2);
		}
		//System.out.println(Math.sqrt(acc));
		return Math.sqrt(acc);
	}

	@Override
	public double classify(Example example) {
		for (ExampleDist ex : pq) {
			if (!pq.contains(new ExampleDist(example, 0))) { // make sure not considering self in k nearest neighbors
				double dist = getEucDistance(example, ex.getExample());
				boolean insert = false; // to be used to determine whether
										// element should be inserted into the
										// priority queue
				double maxDist = getMaxDist(pq);
				if (dist < maxDist) { 
					if (pq.size() >= k) { //an element will need to be kicked out of the priority queue
						replace(new ExampleDist(example, dist));
					} else { //add the new ExampleDist to the priority queue
						pq.add(new ExampleDist(example, dist));
					}
				}
			}
		}		
		return majLabel(pq);
	}
	
	/**
	 * Calculate the majority label contained in the priority queue
	 * @param p - priority queue containing the k nearest neighbors of the example to be classified
	 * @return label which will classify the example
	 */
	protected double majLabel(PriorityQueue<ExampleDist> p) {
		HashMapCounter<Double> counter = new HashMapCounter<Double>();
		for(ExampleDist ex : p) {
			 //TODO figure out how to get majority label
			counter.increment(ex.getExample().getLabel());	
		}
		return counter.sortedEntrySet().get(0).getKey();
	}

	/**
	 * obtain the highest distance currently contained in the priority queue
	 * 
	 * @param p
	 *            - current priority queue
	 * @return maximum distance
	 */
	protected double getMaxDist(PriorityQueue<ExampleDist> p) {
		ArrayList<Double> pqDist = new ArrayList<Double>();
		for (ExampleDist e : pq) {
			pqDist.add(e.getDistance());
		}
		return Collections.max(pqDist);
	}

	/**
	 * Remove the element in the priority queue with the smallest priority, and
	 * add in a new element. This method's purpose is to keep the priority queue
	 * at fixed size k.
	 * 
	 * @param newElt - element to be added to the priority queue
	 */
	protected void replace(ExampleDist newElt) {
		PriorityQueue<ExampleDist> copy = new PriorityQueue<ExampleDist>();
		while (pq.size() > 1) {
			copy.add(pq.poll());
		}
		pq.clear();
		copy.add(newElt);
		pq = copy;
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
