#!/bi/sh
CP=../../treelstm/data/stanford-corenlp-full-2015-12-09/*
java -cp "$CP" -Xmx2g edu.stanford.nlp.pipeline.StanfordCoreNLP \
 -annotators tokenize,ssplit,pos,lemma,ner,parse,dcoref \
 -file sample.input \
 -outputFormat text
