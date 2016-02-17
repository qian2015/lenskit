package org.lenskit.mf.hmmsvd;

import org.junit.Test;

import java.io.*;

import static org.junit.Assert.assertThat;

/**
 * @author <a href="http://www.grouplens.org">GroupLens Research</a>
 */
public class HmmModelBuildTest {
    @Test
    public void testModelBuild() throws FileNotFoundException, IOException {
        String train = "/home/qian/Study/pyml/NoisyNegativeImplicitFeedback/data/hmmsvd11-withlab-clkrat.te";
        HmmSVDFeatureInstanceDAO trDao = new HmmSVDFeatureInstanceDAO(new File(train), " ");
        HmmModelBuilder modelBuilder = new HmmModelBuilder(24, trDao);
        HmmModel model = modelBuilder.build();
        String modelFile = "/home/qian/Study/pyml/NoisyNegativeImplicitFeedback/data/hmm11-withlab-clkrat.model2";
        ObjectOutputStream fout = new ObjectOutputStream(new FileOutputStream(modelFile));
        fout.writeObject(model);
        fout.close();
    }
}
