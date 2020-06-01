#!/usr/bin/env bash

echo "Concatenating files..."
cat text.out knowledge_styletip.out knowledge_attribute.out knowledge_celebrity.out > all_text.out

echo "Spliting files..."
python tools/split.py all_text.out

echo "Converting xml..."
python tools/convert.py src text true pred all_text.out.true
python tools/convert.py ref text true pred all_text.out.true
python tools/convert.py tst text true pred all_text.out.pred

echo "Evaluating..."
perl tools/mteval-v14.pl -s text_src.xml -r text_ref.xml -t text_tst.xml
