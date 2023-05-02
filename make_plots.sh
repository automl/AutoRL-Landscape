#!/bin/bash
algo=$1

phases ana maps data/$algo.csv ilm --savefig
phases ana maps data/$algo.csv igpr --savefig
phases ana modalities data/$algo.csv --savefig
phases ana crashes data/$algo.csv --savefig
phases ana graphs data/$algo.csv ilm --savefig
phases ana graphs data/$algo.csv igpr --savefig

# # --cherrypick "[]" just selects all plots
# phases ana maps data/$algo.csv ilm --cherrypick "[]" --savefig
# phases ana maps data/$algo.csv igpr --cherrypick "[]" --savefig
# phases ana modalities data/$algo.csv --cherrypick "[]" --savefig
# phases ana crashes data/$algo.csv --cherrypick "[]" --savefig
# phases ana graphs data/$algo.csv ilm --cherrypick "[]" --savefig
# phases ana graphs data/$algo.csv igpr --cherrypick "[]" --savefig
