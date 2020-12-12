package br.ufrn.imd.ebssb.core;

import weka.core.Instance;

public class MyInstance {

	private Instance instance;
	private Double weight;
	private Double instanceClass;
	
	public MyInstance(Instance instance) {
		this.instance = instance;
		this.instanceClass = -1.0;
	}
	
	public MyInstance(Instance instance, Double wieght) {
		this.instance = instance;
		this.instanceClass = -1.0;
		this.weight = weight;
	}

	public Instance getInstance() {
		return instance;
	}

	public void setInstance(Instance instance) {
		this.instance = instance;
	}

	public Double getWeight() {
		return weight;
	}

	public void setWeight(Double weight) {
		this.weight = weight;
	}

	public Double getInstanceClass() {
		return instanceClass;
	}

	public void setInstanceClass(Double instanceClass) {
		this.instanceClass = instanceClass;
	}
	
}
