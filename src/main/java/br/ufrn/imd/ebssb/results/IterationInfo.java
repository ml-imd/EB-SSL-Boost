package br.ufrn.imd.ebssb.results;

public class IterationInfo {
	
	private int addedTolabeled;
	private int missClassifiedInstances;
	
	public IterationInfo(int addedTolabeled, int missClassifiedInstances) {
		this.addedTolabeled = addedTolabeled;
		this.missClassifiedInstances = missClassifiedInstances;
	}
	
	public int getAddedTolabeled() {
		return addedTolabeled;
	}

	public void setAddedTolabeled(int addedTolabeled) {
		this.addedTolabeled = addedTolabeled;
	}

	public int getMissClassifiedInstances() {
		return missClassifiedInstances;
	}

	public void setMissClassifiedInstances(int missClassifiedInstances) {
		this.missClassifiedInstances = missClassifiedInstances;
	}
	
	
}
