package ml.utils;

import ml.data.Example;
import ml.data.*;

/**
 * New data structure for k-NN classification, associating all training 
 * examples with a distance from the example being classified.
 *  
 * @author Maddie Gordon, Nick Reminder
 *
 */
public class ExampleDist implements Comparable<ExampleDist> {
	
	Example ex;
	double dist;
	
	public ExampleDist(Example ex, double dist){
		this.ex = ex;
		this.dist = dist;
	}

	@Override
	public int compareTo(ExampleDist other) {
		// TODO Auto-generated method stub
		return Double.compare(dist, other.getDistance());
	} 
	
	public Example getExample() {
		return ex;
	}
	
	public double getDistance(){
		return dist;
	}

}
