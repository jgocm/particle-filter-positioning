# particle-filter-positioning
Estimating optimal positions in the soccer field using particle filters

## implement in unification

### currently we have:
map of positions -> compute likelihoods according to metrics -> rank -> assign positions with hungarian

### would change to:
update particle filter -> add output to possible positions -> map of positions -> compute likelihoods according to metrics -> rank -> assign positions with hungarian