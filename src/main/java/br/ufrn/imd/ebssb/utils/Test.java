package br.ufrn.imd.ebssb.utils;

import br.ufrn.imd.ebssb.core.Dataset;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.filters.unsupervised.attribute.RandomSubset;

public class Test {

	
	public static void main(String[] args) throws Exception {
		Dataset d = new Dataset("src/main/resources/datasets/experiment2/"+"Car.arff");
		Instance[] centroids = Mathematics.centroidsOf(d.getInstances());
		
		for(int i = 0; i < centroids.length; i++) {
			System.out.println("centroid " + i +" = " + centroids[i].toString());
		}
		
		System.out.println("DISTANCES");
		
		Instance inst = (Instance) d.getInstances().get(0).copy();
		
		for(int i = 0; i < centroids.length; i++) {
			System.out.println(Mathematics.euclidianDistance(inst, centroids[i]));
		}
		
		RandomSubset rss = new RandomSubset();
		rss.setInputFormat(d.getInstances());
		System.out.println(rss.getOutputFormat());
		
	}
	
	
}
