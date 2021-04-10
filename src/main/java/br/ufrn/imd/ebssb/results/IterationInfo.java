package br.ufrn.imd.ebssb.results;

public class IterationInfo {

	private String testSetSummary;
	private String boostSubsetSummary;

	private int instancesSampledAndLabelledByPool;
	private int boostSubSetSize;
	private int boostEnsembleHits;
	private int boostEnsembleErrors;

	private int labelledInstances;
	private int unlabelledInstances;

	public IterationInfo() {

	}

	public String getTestSetSummary() {
		return testSetSummary;
	}

	public void setTestSetSummary(String testSetSummary) {
		this.testSetSummary = testSetSummary;
	}

	public String getBoostSubsetSummary() {
		return boostSubsetSummary;
	}

	public void setBoostSubsetSummary(String boostSubsetSummary) {
		this.boostSubsetSummary = boostSubsetSummary;
	}

	public int getInstancesSampledAndLabelledByPool() {
		return instancesSampledAndLabelledByPool;
	}

	public void setInstancesSampledAndLabelledByPool(int instancesSampledAndLabelledByPool) {
		this.instancesSampledAndLabelledByPool = instancesSampledAndLabelledByPool;
	}

	public int getBoostEnsembleHits() {
		return boostEnsembleHits;
	}

	public void setBoostEnsembleHits(int boostEnsembleHits) {
		this.boostEnsembleHits = boostEnsembleHits;
	}

	public int getBoostEnsembleErrors() {
		return boostEnsembleErrors;
	}

	public void setBoostEnsembleErrors(int boostEnsembleErrors) {
		this.boostEnsembleErrors = boostEnsembleErrors;
	}

	public int getUnlabelledInstances() {
		return unlabelledInstances;
	}

	public void setUnlabelledInstances(int unlabelledInstances) {
		this.unlabelledInstances = unlabelledInstances;
	}

	public int getBoostSubSetSize() {
		return boostSubSetSize;
	}

	public void setBoostSubSetSize(int boostSubSetSize) {
		this.boostSubSetSize = boostSubSetSize;
	}

	public int getLabelledInstances() {
		return labelledInstances;
	}

	public void setLabelledInstances(int labelledInstances) {
		this.labelledInstances = labelledInstances;
	}

	public String getIterationInfoSummary() {

		StringBuilder sb = new StringBuilder();
		sb.append("\n-------------------------------\n");
		sb.append("Labelled instances amount: " + this.labelledInstances + "\n");
		sb.append("Unabelled instances amount: " + this.unlabelledInstances + "\n");
		sb.append("Sampled instances that had a high agreement: " + this.instancesSampledAndLabelledByPool + "\n");
		sb.append("\n");
		sb.append("BoostSubSet size: " + this.boostSubSetSize + "\n");
		sb.append("Current Boost ensemble hits: " + this.boostEnsembleHits + "\n");
		sb.append("Current Boost ensemble errors: " + this.boostEnsembleErrors + "\n");
		sb.append("-------------------------------\n");
		return sb.toString();
	}
	
	public String onlyValuesToString() {
		StringBuilder sb = new StringBuilder();
		sb.append(this.labelledInstances + "\t\t");
		sb.append(this.unlabelledInstances + "\t\t");
		sb.append(this.instancesSampledAndLabelledByPool + "\t\t");
		sb.append(this.boostSubSetSize + "\t\t");
		sb.append(this.boostEnsembleHits + "\t\t");
		sb.append(this.boostEnsembleErrors + "\t\t");
		return sb.toString();
	}

}
