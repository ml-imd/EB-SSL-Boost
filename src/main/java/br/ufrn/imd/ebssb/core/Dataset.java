package br.ufrn.imd.ebssb.core;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Dataset {

	private Instances instances;

	private String datasetName;
	private ArrayList<MyInstance> myInstances;
	private double totalWeight;

	private HashMap<String, Integer> positions;

	public Dataset() {

	}

	public Dataset(String pathAndDataSetName) throws Exception {
		instances = DataSource.read(pathAndDataSetName);
		instances.setClassIndex(instances.numAttributes() - 1);
		this.datasetName = instances.relationName();
		this.totalWeight = -1.0;

		matchInstancesAndMyInstances();

		this.positions = new HashMap<String, Integer>();
	}

	public Dataset(Instances instances) {
		this.instances = new Instances(instances);
		this.datasetName = instances.relationName();
		this.totalWeight = -1.0;

		matchInstancesAndMyInstances();

		this.positions = new HashMap<String, Integer>();
	}

	public Dataset(Dataset dataset) {
		this.instances = new Instances(dataset.getInstances());
		this.datasetName = instances.relationName();
		this.totalWeight = -1.0;

		matchInstancesAndMyInstances();

		this.positions = new HashMap<String, Integer>();
	}

	public void shuffleInstances(int seed) {
		this.instances.randomize(new Random(seed));

		matchInstancesAndMyInstances();
	}

	public void addInstance(Instance instance) {
		Instance a = instance;
		this.instances.add(a);

		this.myInstances.add(new MyInstance(a));
	}

	public void addMyInstance(MyInstance myInstance) {

		this.instances.add(myInstance.getInstance());

		this.myInstances.add(myInstance);
		increaseTotalWeight(myInstance.getWeight());
	}

	public void addLabelledInstance(Instance instance) {
		Instance a = instance;
		this.instances.add(a);

		MyInstance myInst = new MyInstance(a);
		myInst.setInstanceClass(a.classValue());
		this.myInstances.add(myInst);
	}

	public void clearInstances() {
		this.instances.clear();

		this.myInstances = new ArrayList<MyInstance>();
		this.totalWeight = 0.0;
		this.positions = new HashMap<String, Integer>();
	}

	public void initInstancesWeight(Double initialWeight) {
		for (MyInstance m : myInstances) {
			m.setWeight(initialWeight);
		}
	}

	public void increaseTotalWeight(double value) {
		this.totalWeight += value;
	}

	private void matchInstancesAndMyInstances() {
		this.myInstances = new ArrayList<MyInstance>();

		for (Instance i : instances) {
			MyInstance m = new MyInstance(i);
			myInstances.add(m);
		}
	}

	public static ArrayList<Dataset> splitDataset(Dataset dataset, int numberOfParts) {

		ArrayList<Dataset> splitedDataset = new ArrayList<Dataset>();
		ArrayList<Instance> myData = new ArrayList<Instance>();
		ArrayList<Instance> part = new ArrayList<Instance>();

		int size = dataset.getInstances().size() / numberOfParts;

		for (Instance i : dataset.getInstances()) {
			myData.add(i);
		}
		int i = 0;
		int control = 0;

		for (i = 0; i < myData.size(); i++) {
			part.add(myData.get(i));
			if (part.size() == size) {
				Dataset d = new Dataset();
				d.setDatasetName(dataset.getInstances().relationName());
				d.setInstances(new Instances(dataset.getInstances()));
				d.getInstances().clear();
				d.getInstances().addAll(part);

				d.matchInstancesAndMyInstances();

				splitedDataset.add(d);
				part = new ArrayList<Instance>();
				control = i;
			}
		}
		int x = 0;

		while (control < (myData.size() - 1)) {
			splitedDataset.get(x).addInstance(myData.get(control));
			control++;
			x++;
			if (x == (splitedDataset.size() - 1)) {
				x = 0;
			}
		}
		return splitedDataset;
	}

	public static Dataset joinDatasets(ArrayList<Dataset> folds) {
		Instances ins = folds.get(0).getInstances();

		for (int i = 1; i < folds.size(); i++) {
			ins.addAll(folds.get(i).getInstances());
		}
		return new Dataset(ins);
	}

	public MyInstance drawOne(MyRandom myRandom) {

		int aux = myRandom.nextInt((int) this.totalWeight);
		MyInstance drawed = new MyInstance();

		for (MyInstance m : this.myInstances) {
			aux -= m.getWeight();
			if (aux < 0) {
				drawed = m;
				break;
			}
		}
		return drawed;
	}

	public String getMyInstancesSummary() {

		StringBuilder sb = new StringBuilder();
		sb.append(
				"[        instance        ]: weight; instanceClass; -> result: {agreement per class}; bestClass; bestResult\n");
		for (MyInstance m : myInstances) {
			sb.append(m.toString());
			sb.append("\n");
		}
		return sb.toString();
	}

	public void storePositions() {
		for(int i = 0; i < this.myInstances.size(); i++) {
			myInstances.get(i).generateHashForInstance();
			this.positions.put(myInstances.get(i).getHashId(), i);
		}
	}
	
	//GETTERS AND SETTERS
	
	public ArrayList<MyInstance> getMyInstances() {
		return myInstances;
	}

	public void setMyInstances(ArrayList<MyInstance> myInstances) {
		this.myInstances = myInstances;
	}

	public Double getTotalWeight() {
		return totalWeight;
	}

	public void setTotalWeight(Double totalWeight) {
		this.totalWeight = totalWeight;
	}

	public Instances getInstances() {
		return instances;
	}

	public void setInstances(Instances instances) {
		this.instances = instances;

		matchInstancesAndMyInstances();
	}

	public String getDatasetName() {
		return datasetName;
	}

	public void setDatasetName(String datasetName) {
		this.datasetName = datasetName;
	}

	public HashMap<String, Integer> getPositions() {
		return positions;
	}

	public void setPositions(HashMap<String, Integer> positions) {
		this.positions = positions;
	}

	
}
