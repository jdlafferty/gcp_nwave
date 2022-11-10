#!/bin/sh

actsim -cnf=x.conf -Wlang_subst:off nn4.act 'grid<8,3,2>' <<EOF
cycle
EOF
