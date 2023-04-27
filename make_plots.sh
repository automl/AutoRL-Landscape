#!/bin/bash

phases ana maps data/ppo.csv ilm --savefig
phases ana maps data/ppo.csv igpr --savefig
phases ana modalities data/ppo.csv --savefig
phases ana crashes data/ppo.csv --savefig
phases ana graphs data/ppo.csv ilm --savefig
phases ana graphs data/ppo.csv igpr --savefig

# --cherrypick "[]" just selects all plots
phases ana maps data/ppo.csv ilm --cherrypick "[]" --savefig
phases ana maps data/ppo.csv igpr --cherrypick "[]" --savefig
phases ana modalities data/ppo.csv --cherrypick "[]" --savefig
phases ana crashes data/ppo.csv --cherrypick "[]" --savefig
phases ana graphs data/ppo.csv ilm --cherrypick "[]" --savefig
phases ana graphs data/ppo.csv igpr --cherrypick "[]" --savefig
