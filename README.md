# Orcinus Bootstrap [![Build Status](https://travis-ci.org/orcinus-lang/orcinus-bootstrap.svg?branch=master)](https://travis-ci.org/orcinus-lang/orcinus-bootstrap)

This repository contains bootstrap version of my hobby compiler (`orcinus` - static Python).

- `bootstrap.py` - this script is used Python AST for emitting LLVM IR

This is only `bootstrap` compiler, read [Self-hosting](https://en.wikipedia.org/wiki/Self-hosting) 
and [Bootstrapping](https://en.wikipedia.org/wiki/Bootstrapping_(compilers)) on Wiki

Capabilities
------------

Not implemented!

Install
-------

Use Python 3.7 and install project dependencies:

```bash
pip install -r requirements.txt
```  

Usage
-----

For compile source code to LLVM IR use next command:

```bash
./bootstrap.py example.orx > example.ll 
```

For execute generated LLVM IR:

```bash
lli-6.0 example.ll
```

Examples
--------

View scripts in directory `tests`.
