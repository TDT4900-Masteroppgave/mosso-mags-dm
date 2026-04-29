package mosso;

import mosso.algorithm.*;

import java.io.IOException;
import java.util.Date;

public class Run {

    private SummaryGraphModule module;

    public static void main(String[] args) throws IOException {
        Date today = new Date();
        System.out.println(today);
        final String inputPath = args[0];
        System.out.println("input_path: " + inputPath);
        final String outputPath = args[1];
        System.out.println("output_path: " + outputPath);
        final String sumMode = args[2];
        System.out.println("summarization_mode: " + sumMode);
        System.out.println();

        final SummaryGraphModule module;

        if (sumMode.compareTo("mosso") == 0) {
            final int probability = Integer.parseInt(args[3]);
            final int n_samples = Integer.parseInt(args[4]);
            final int interval = Integer.parseInt(args[5]);
            final double threshold_start = Double.parseDouble(args[6]);
            final double threshold_end = Double.parseDouble(args[7]);
            final int num_iterations = Integer.parseInt(args[8]);
            System.out.println("escape probability : " + probability + ", n_samples : " + n_samples + ", TT : " + interval + ", threshold_start : " + threshold_start + ", threshold_end : " + threshold_end + ", num_iterations : " + num_iterations);
            module = new MoSSo(false, probability, n_samples, interval, threshold_start, threshold_end, num_iterations);
        } else if (sumMode.compareTo("simple") == 0) {
            final int probability = Integer.parseInt(args[3]);
            final int n_samples = Integer.parseInt(args[4]);
            final int interval = Integer.parseInt(args[5]);
            System.out.println("escape probability : " + probability + ", n_samples : " + n_samples + ", TT : " + interval);
            module = new MoSSoSimple(false, probability, n_samples, interval);
        } else if (sumMode.compareTo("mcmc") == 0) {
            final int interval = Integer.parseInt(args[3]);
            System.out.println("interval : " + interval);
            module = new MoSSoMCMC(false, interval);
        } else if(sumMode.compareTo("sgreedy") == 0) {
            final int interval = Integer.parseInt(args[3]);
            System.out.println("interval : " + interval);
            module = new MoSSoGreedy(false, interval);
        } else {
            System.out.println("Invalid command.");
            return;
        }
        int edgeCount = Common.execute(module, inputPath, "\t");
        Common.writeOutputs(module, "output/" + outputPath, edgeCount);
    }
}
