# xWRF
 A simple command-line tool for extracting electron temperatures from wedge image plate data
 To use, call the command
 ```bash
 python analyze_xwrf.py SHOT_NUMBER WEDGE_ID [--filter=FILTER_THICKNESS] [--nose] [--cr39]
 ```
 where `SHOT_NUMBER` is any substring that uniquely identifies the file you want to analyze (most
 likely the shot number) and `WEDGE_ID` is the ID of the wedge range filter (for example, "G069").
 If you fielded the image plate with more filtering than just the wedge, then specify it in the
 optional arguments.  Using `--filter` adds a flat aluminium filter; specify the thickness in
 micrometers by replacing `FILTER_THICKNESS`.  Using `--nose` adds an aluminium nose cap and is
 equivalent to `--filter=300` (note that they do stack, so feel free to use both).  Using `--cr39`
 adds a standard 1500 μm piece of CR-39.

 The program will print out the result and show you some plots of the data so that you can judge
 for yourself that it found the fiducials and fit the data correctly.

 If you have multiple shots to analyze, feel free to edit `analyze_all_xwrfs.py` and then simply call
 ```bash
 python analyze_all_xwrfs.py
 ```
 I think that script is pretty self-explanatory.

# To do
 Error bars.
