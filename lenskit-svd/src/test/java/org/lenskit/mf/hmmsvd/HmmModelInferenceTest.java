package org.lenskit.mf.hmmsvd;

import org.apache.commons.lang3.StringUtils;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.StatUtils;
import org.junit.Test;

import java.io.*;
import java.util.ArrayList;

/**
 * @author <a href="http://www.grouplens.org">GroupLens Research</a>
 */
public class HmmModelInferenceTest {
    @Test
    public void testModelInference() throws ClassNotFoundException, IOException {
        //String modelFile = "/home/qian/Study/pyml/NoisyNegativeImplicitFeedback/data/explore11-withlab-clkrat-feas.hmm.tr.merge.hmm";
        String modelFile = "/home/qian/Study/pyml/NoisyNegativeImplicitFeedback/data/hmm11-withlab-clkrat.est.merge.model";
        String testFile = "/home/qian/Study/pyml/NoisyNegativeImplicitFeedback/data/hmmsvd11-withlab-clkrat.te";
        String predFile = "/home/qian/Study/pyml/NoisyNegativeImplicitFeedback/data/hmmsvd11-withlab-clkrat.te.pred";

        ObjectInputStream fin = new ObjectInputStream(new FileInputStream(modelFile));
        HmmModel model = (HmmModel)(fin.readObject());

        //HmmModel model = new HmmModel(24);
        //model.loadFromTextFile(new File(modelFile));

        HmmSVDFeatureInstanceDAO teDao = new HmmSVDFeatureInstanceDAO(new File(testFile), " ");
        BufferedWriter fout = new BufferedWriter(new FileWriter(predFile));
        HmmSVDFeatureInstance ins = teDao.getNextInstance();
        while(ins != null) {
            ArrayList<RealVector> gamma = new ArrayList<>(ins.numObs);
            ArrayList<ArrayList<RealVector>> xi = new ArrayList<>(ins.numObs - 1);
            ArrayList<RealVector> probX = new ArrayList<>(ins.numObs);
            model.inference(ins, gamma, xi, probX);
            RealVector sumVec = MatrixUtils.createRealVector(new double[ins.numPos]);
            for (int i=0; i<ins.numObs; i++) {
                sumVec.combineToSelf(1.0, 1.0, gamma.get(i));
            }
            double sum = StatUtils.sum(((ArrayRealVector)sumVec).getDataRef());
            String[] line1 = new String[ins.numPos];
            String[] line2 = new String[ins.numPos];
            for (int i=0; i<ins.numPos; i++) {
                line1[i] = Double.toString(ins.numObs * sumVec.getEntry(i) / sum);
                line2[i] = Double.toString(sumVec.getEntry(i));
            }
            fout.write(StringUtils.join(line1, " ") + ":" + StringUtils.join(line2, " ") + "\n");
            ins = teDao.getNextInstance();
        }
        fout.close();
    }
}
