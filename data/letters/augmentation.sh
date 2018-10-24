CHARS=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ

pdftoppm letters.pdf out -png

rm out-01.png

for (( i=0; i<8; i++ )); do
    mkdir "${CHARS:$i:1}"
    mv out-0$(($i + 2)).png "${CHARS:$i:1}/O.png"
done

for (( i=8; i<${#CHARS}; i++ )); do
    mkdir "${CHARS:$i:1}"
    mv out-$(($i + 2)).png "${CHARS:$i:1}/O.png"
done
