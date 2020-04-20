mkdir -p data
cd data

W2VFIRST='https://docs.google.com/uc?export=download'
W2VSECOND='https://docs.google.com/uc?export=download&confirm='
W2VID='&id=0B7XkCwpI5KDYNlNUTTlSS21pQmM'
W2VGZ='GoogleNews-vectors-negative300.bin.gz'
W2VFILE='GoogleNews-vectors-negative300.bin'

if [ ! -e $W2VFILE ]; then
    if [ ! -e $W2VGZ ]; then
        OUTPUT=$( wget --save-cookies cookies.txt --keep-session-cookies --no-check-certificate $W2VFIRST$W2VID  -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/Code: \1\n/p' )
        CODE=${OUTPUT##*Code: }
        wget --load-cookies cookies.txt $W2VSECOND$CODE$W2VID -O $W2VGZ
        rm cookies.txt
    fi
    gunzip -dc $W2VGZ > $W2VFILE
fi


GLOVEURL="http://nlp.stanford.edu/data/glove.6B.zip"
GLOVEZIP="glove.6B.zip"
GLOVEFILE="glove.6B.100d.txt"

if [ ! -e $GLOVEFILE ]; then
    if [ ! -e $GLOVEZIP ]; then
        wget $GLOVEURL
    fi
    unzip -o $GLOVEZIP
fi
