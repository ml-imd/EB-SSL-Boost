package br.ufrn.imd.ebssb.core;

import java.util.ArrayList;

import br.ufrn.imd.ebssb.results.InstanceResult;
import weka.core.Instance;

public class MyInstance {

	private Instance instance;
	private Double weight;
	private Double instanceClass;
	private ArrayList<InstanceResult> results;
	
	public MyInstance(Instance instance) {
		this.instance = instance;
		this.instanceClass = -1.0;
		this.results = new ArrayList<InstanceResult>();
	}
	
	public MyInstance(Instance instance, Double weight) {
		this.instance = instance;
		this.instanceClass = -1.0;
		this.weight = weight;
		this.results = new ArrayList<InstanceResult>();
	}

	public void increaseWeight(Double value) {
		this.weight += value;
	}
	
	public void decreaseWeight(Double value) {
		this.weight -= value;
	}
	
	public void addInstanceResult(InstanceResult result) {
		this.results.add(result);
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

	public ArrayList<InstanceResult> getResults() {
		return results;
	}

	public void setResults(ArrayList<InstanceResult> results) {
		this.results = results;
	}
	
}
