# Short reaction intermissions

The `.b64` files contain the original small image/GIF bytes as base64 text. The
pipeline validates their checksums and decodes them locally, so making a video
does not depend on a meme website or image API being online.

| Asset | Reaction | Source |
| --- | --- | --- |
| `self-high-five.gif.b64` | Liz Lemon's self high five, 23 moving frames | [milsyobtaf/gifs](https://github.com/milsyobtaf/gifs/blob/primary/30rock-self-highfive.gif) |
| `spock-cat.gif.b64` | Spock stroking a cat, 30 moving frames | [tlberglund/animated-gifs](https://github.com/tlberglund/animated-gifs/blob/master/spock-stroking-cat.gif) |
| `crying-thumbs-up-cat.jpg.b64` | Crying thumbs-up cat, still photo | [Pratikshya32/cat-meme-cam, `memes.js` `thumbsup`](https://github.com/Pratikshya32/cat-meme-cam/blob/main/memes.js) |

The GIFs retain their original animation. Square and portrait reactions fit within
a soft duplicate backdrop rather than having faces or hands cropped off. Spoken
jokes are selected together with their reactions; a missing asset uses another
matching pair or a movie-background message.

Source credits are provenance, not an assertion that the source repositories own
or license the underlying television footage or meme photographs. Original
rights remain with their respective owners. Replace the curated assets with
channel-owned reactions if those are required for a particular publication.
