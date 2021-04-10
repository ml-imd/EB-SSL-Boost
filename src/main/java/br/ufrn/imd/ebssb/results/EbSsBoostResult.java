package br.ufrn.imd.ebssb.results;

import java.util.ArrayList;

import br.ufrn.imd.ebssb.utils.DateUtils;
import br.ufrn.imd.ebssb.utils.NumberUtils;

public class EbSsBoostResult {

	private int numFolds;
	private String datasetName;
	private String selfTrainingVersion;
	private ArrayList<FoldResult> results;
	private FoldResult averageResult;
	private long begin;
	private long end;

	public EbSsBoostResult(int numFolds, String datasetName, String EbSsBoostVersion) {
		this.numFolds = numFolds;
		this.datasetName = new String(datasetName);
		this.selfTrainingVersion = new String(EbSsBoostVersion);
		this.results = new ArrayList<FoldResult>();
		this.averageResult = new FoldResult();
	}

	public void addFoldResult(FoldResult result) {

		if (results.size() < numFolds) {
			this.results.add(result);
			if (results.size() == numFolds) {
				calcAverageResult();
			}
		} else {
			System.out.println("result from this dataset is already full");
		}
	}

	public void calcAverageResult() {

		int num = results.size();

		double accuracy = 0.0;
		double error = 0.0;
		double fMeasure = 0.0;
		double precision = 0.0;
		double recall = 0.0;

		for (FoldResult fr : results) {
			accuracy += fr.getAccuracy();
			error += fr.getError();
			fMeasure += fr.getfMeasure();
			precision += fr.getPrecision();
			recall += fr.getRecall();
		}

		averageResult.setAccuracy(accuracy / num);
		averageResult.setError(error / num);
		averageResult.setfMeasure(fMeasure / num);
		averageResult.setPrecision(precision / num);
		averageResult.setRecall(recall / num);
	}

	public void showResult() {
		System.out.println(buildResultString());
	}

	public String getResult() {
		return buildResultString();
	}

	private String buildMetrics() {
		StringBuilder sb = new StringBuilder();
		sb.append("------------------------------------------------------------------------------------------------");
		sb.append("\n");
		sb.append("@DATASET: " + datasetName + "\n");
		sb.append("@Folds  : " + numFolds + "\n");
		sb.append("@STvers : " + selfTrainingVersion + "\n");
		sb.append("------------------------------------------------------------------------------------------------");
		sb.append("\n\t\t");
		sb.append(" accura " + "\t");
		sb.append("  error " + "\t");
		sb.append(" fmeasu " + "\t");
		sb.append(" precis " + "\t");
		sb.append(" recall " + "\t\n");
		sb.append("------------------------------------------------------------------------------------------------");
		sb.append("\n");
		for (int i = 0; i < results.size(); i++) {
			if (i < 9) {
				sb.append("fold0" + (i + 1) + ": \t");
				sb.append(results.get(i).onlyValuesToString() + "\n");
			} else {
				sb.append("fold" + (i + 1) + ": \t");
				sb.append(results.get(i).onlyValuesToString() + "\n");
			}
		}
		sb.append("------------------------------------------------------------------------------------------------");
		sb.append("\n");
		sb.append("AVERAGE" + "\t\t");
		sb.append(averageResult.onlyValuesToString() + "\n");
		sb.append("------------------------------------------------------------------------------------------------");
		sb.append("\n");
		return sb.toString();
	}

	private String buildTime() {
		StringBuilder sb = new StringBuilder();
		sb.append("------------------------------------------------------------------------------------------------");
		sb.append("\n");
		sb.append("BEGIN: \t" + DateUtils.fromLongToDateAsString(this.begin));
		sb.append("\n");
		sb.append("END: \t" + DateUtils.fromLongToDateAsString(this.end));
		sb.append("\n");
		sb.append("\n");
		sb.append("TIME ELAPSED:\t" + NumberUtils.doubleToString(getTimeElapsed()) + " SECONDS");
		sb.append("\n");
		sb.append("------------------------------------------------------------------------------------------------");
		sb.append("\n");
		return sb.toString();
	}

	private String buildResultString() {
		StringBuilder sb = new StringBuilder();
		sb.append(buildMetrics());
		sb.append(buildTime());
		return sb.toString();
	}

	public long getTimeElapsed() {
		return (this.end - this.begin) / 1000;
	}

	public int getNumFolds() {
		return numFolds;
	}

	public void setNumFolds(int numFolds) {
		this.numFolds = numFolds;
	}

	public String getDatasetName() {
		return datasetName;
	}

	public void setDatasetName(String datasetName) {
		this.datasetName = datasetName;
	}

	public String getSelfTrainingVersion() {
		return selfTrainingVersion;
	}

	public void setSelfTrainingVersion(String selfTrainingVersion) {
		this.selfTrainingVersion = selfTrainingVersion;
	}

	public ArrayList<FoldResult> getResults() {
		return results;
	}

	public void setResults(ArrayList<FoldResult> results) {
		this.results = results;
	}

	public FoldResult getAverageResult() {
		return averageResult;
	}

	public void setAverageResult(FoldResult averageResult) {
		this.averageResult = averageResult;
	}

	public long getBegin() {
		return begin;
	}

	public void setBegin(long begin) {
		this.begin = begin;
	}

	public long getEnd() {
		return end;
	}

	public void setEnd(long end) {
		this.end = end;
	}

}
