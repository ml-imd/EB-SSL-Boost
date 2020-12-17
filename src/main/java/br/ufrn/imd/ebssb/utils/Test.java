package br.ufrn.imd.ebssb.utils;

import br.ufrn.imd.ebssb.core.Dataset;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.filters.unsupervised.attribute.RandomSubset;

public class Test {

	
	public static void main(String[] args) throws Exception {
		
		int agreementValue = 22 * 75 / 100;
		int value = (int) agreementValue;
		
		System.out.println(agreementValue);
		System.out.println(value);
		
	}
	
	
}
