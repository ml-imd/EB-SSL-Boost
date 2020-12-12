package br.ufrn.imd.ebssb.utils;

import java.util.LinkedList;

import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;


/**
 * 
 * @author José Gameleira do Rêgo Neto
 * @email jgdrneto@gmail.com
 * 
 * @see <a href="https://github.com/jgdrneto/AvaliacaoApredizadoMaquina">AvaliacaoApredizadoMaquina
 *
 */
public class Mathematics {

	/** Method for computing deviation of double values */
	public static double deviation(double[] x) {
		double mean = mean(x);
		double squareSum = 0;

		for (int i = 0; i < x.length; i++) {
			squareSum += Math.pow(x[i] - mean, 2);
		}

		return Math.sqrt((squareSum) / (x.length - 1));
	}

	/** Method for computing deviation of int values */
	public static double deviation(int[] x) {
		double mean = mean(x);
		double squareSum = 0;

		for (int i = 0; i < x.length; i++) {
			squareSum += Math.pow(x[i] - mean, 2);
		}

		return Math.sqrt((squareSum) / (x.length - 1));
	}

	/** Method for computing mean of an array of double values */
	public static double mean(double[] x) {
		double sum = 0;

		for (int i = 0; i < x.length; i++)
			sum += x[i];

		return sum / x.length;
	}

	/** Method for computing mean of an array of int values */
	public static double mean(int[] x) {
		int sum = 0;

		for (int i = 0; i < x.length; i++)
			sum += x[i];

		return sum / x.length;
	}

	public static long combinationOf(long a, long b) {
		if ((a == b) || (b == 0)){
			return 1;
		} else if(b == (a-1)){
			return a;
		}
		
		if(b > (a-b)){
			b = Math.abs(a-b);
		}
		
		return arrange(a, b) / factorial(b);
	}

	public static long factorial(long n) {
		if (n < 3) {
			return n;
		}

		long x = 2;
		for (long i = 3; i <= n; i++) {
			x *= i;
		}

		return x;
	}

	public static long arrange(long a, long b) {
		if ((a == b) || (b == 0))
			return 1;

		long x = a;
		for (long i = x - 1; i >= (a - b + 1); i--) {
			x *= i;
		}
		return x;
	}

	public static double clusterDiameter(Instances base,
			DistanceFunction distance, double clusterID) {
		int numInstances = base.numInstances();

		double maxDist = Double.MIN_NORMAL;
		double dist;
		Instance instA, instB;
		for (int i = 0; i < numInstances; i++) {
			for (int j = i + 1; j < numInstances; j++) {
				instA = base.instance(i);
				instB = base.instance(j);

				if ((instA.classValue() == instB.classValue())
						&& (instA.classValue() == clusterID)) {
					dist = distance
							.distance(base.instance(i), base.instance(j));
					if (dist > maxDist) {
						maxDist = dist;
					}
				}
			}
		}

		return maxDist;
	}
	
	public static double maxClusterDiameter(Instances base, DistanceFunction distance){
		double maxDiam = Double.MIN_VALUE;
		
		int numClusters = base.numClasses();
		
		double diam;
		for (int i = 0; i < numClusters; i++) {
			diam = clusterDiameter(base, distance, i);
			if(diam > maxDiam){
				maxDiam = diam;
			}
		}
		
		return maxDiam;
	}

	public static double clusterDissimilarity(Instances base,
			DistanceFunction distance, double clusterIDA, double clusterIDB) {
		int numInstances = base.numInstances();

		double minDist = Double.MAX_VALUE;
		double dist;
		Instance instA, instB;
		for (int i = 0; i < numInstances; i++) {
			for (int j = i + 1; j < numInstances; j++) {
				instA = base.instance(i);
				instB = base.instance(j);

				if (((instA.classValue() == clusterIDA) && (instB.classValue() == clusterIDB))
						|| ((instA.classValue() == clusterIDB) && (instB
								.classValue() == clusterIDA))) {
					dist = distance
							.distance(base.instance(i), base.instance(j));
					if (dist < minDist) {
						minDist = dist;
					}
				}
			}
		}

		return minDist;
	}

	public static double totalBetweenClusterVariance(Instances base,
			DistanceFunction distance) {
		Instance[] centroids = centroidsOf(base);
		Instance center = centerOf(base);

		double result = 0;

		for (Instance centroid : centroids) {
			result += Math.pow(distance.distance(centroid, center), 2);
		}

		return result;
	}

	public static double totalWithinClusterVariationOfCluster(Instances base,
			DistanceFunction distance) {
		int numGrupos = base.numClasses();
		double resultado = 0;

		for (int i = 0; i < numGrupos; i++) {
			resultado += withinClusterVariationOfCluster(base, i, distance);
		}

		return resultado;
	}

	public static double withinClusterVariationOfCluster(Instances base,
			int clusterId, DistanceFunction distance) {
		Instance centroid = centroidOf(base, clusterId);
		double resultado = 0;

		Instance object;

		LinkedList<Integer> objetosNoCluster = objectsFromCluster(base,
				clusterId);

		for (Integer objeto : objetosNoCluster) {
			object = base.instance(objeto);
			resultado += Math.pow(distance.distance(object, centroid), 2);
		}

		return resultado;
	}

	private static Instance centroidOf(Instances base, int clusterId) {
		Instance centroid = new SparseInstance(base.instance(0));
		centroid.setDataset(base);
		int numAttributes = centroid.numAttributes();

		double accumulator;
		Instance object;

		LinkedList<Integer> objetosNoCluster = objectsFromCluster(base,
				clusterId);

		int numObjetos = objetosNoCluster.size();

		for (int i = 0; i < numAttributes; i++) {
			accumulator = 0;
			for (Integer objeto : objetosNoCluster) {
				object = base.instance(objeto);
				accumulator += object.value(i);
			}
			centroid.setValue(i, accumulator / numObjetos);
		}

		return centroid;
	}

	public static Instance centerOf(Instances base) {
		Instance centroid = new SparseInstance(base.instance(0));
		int numAttributes = centroid.numAttributes();

		double accumulator;
		Instance object;

		int numObjetos = base.numInstances();

		for (int i = 0; i < numAttributes; i++) {
			accumulator = 0;
			for (int objeto = 0; objeto < numObjetos; objeto++) {
				object = base.instance(objeto);
				accumulator += object.value(i);
			}
			centroid.setValue(i, accumulator / numObjetos);
		}

		return centroid;
	}

	public static Instance[] centroidsOf(Instances base) {
		int numGrupos = base.numClasses();
		Instance[] centroids = new Instance[numGrupos];

		for (int i = 0; i < centroids.length; i++) {
			centroids[i] = centroidOf(base, i);
		}

		return centroids;
	}

	private static LinkedList<Integer> objectsFromCluster(Instances base,
			int clusterId) {
		int numInstances = base.numInstances();
		Instance object;

		LinkedList<Integer> objetosNoCluster = new LinkedList<Integer>();

		for (int i = 0; i < numInstances; i++) {
			object = base.instance(i);
			if (object.classValue() == ((double) clusterId)) {
				objetosNoCluster.add(i);
			}
		}

		return objetosNoCluster;

	}

	public static double[][] baseDistanceMatrix(Instances instances,
			DistanceFunction distance, int clusterID) {
		int numInstances = 0;
		int aux = instances.numInstances();

		for (int i = 0; i < aux; i++) {
			if (instances.instance(i).classValue() == ((double) clusterID)) {
				numInstances++;
			}
		}

		double[][] distanceMatrix = new double[numInstances][numInstances];

		int ii = 0, jj = 0;
		for (int i = 0; i < aux; i++) {
			for (int j = i; j < aux; j++) {
				if (instances.instance(i).classValue() == instances.instance(j)
						.classValue()) {
					distanceMatrix[ii][jj] = distance.distance(
							instances.instance(i), instances.instance(j));
					distanceMatrix[jj][ii] = distanceMatrix[ii][jj];
					jj++;

					if (jj == numInstances) {
						jj = 0;
						ii++;
						if (ii == numInstances) {
							ii = 0;
						}
					}
				}
			}
		}

		return distanceMatrix;
	}

	//Euclidian Distance
	public static double euclidianDistance(Instance x, Instance y) {
		
		if(x==null || y==null) {
			return Double.POSITIVE_INFINITY;
		}
		
		if (x.numAttributes() != y.numAttributes())
	        throw new IllegalArgumentException(String.format("Arrays have different length: x[%d], y[%d]", x.numAttributes(), y.numAttributes()));

	    int n = x.numAttributes();
	    double dist = 0.0;
	    for (int i = 0; i < n; i++) {
	    	if (!Double.isNaN(x.value(i)) && !Double.isNaN(y.value(i))) {
	            double d = x.value(i) - y.value(i);
	            dist += d * d;
	        }
	    }	    
	    return Math.sqrt(dist);
	}
	
	
}
