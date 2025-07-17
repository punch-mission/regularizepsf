5# Changelog

[This is available in GitHub](https://github.com/punch-mission/regularizepsf/releases?page=1)

## Version 1.1.0: Jul 17, 2025

* specify codecov path by @jmbhughes in https://github.com/punch-mission/regularizepsf/pull/234
* allow a single star mask by @svank in https://github.com/punch-mission/regularizepsf/pull/236
* handle saturation carefully when applying a transform by @jmbhughes in https://github.com/punch-mission/regularizepsf/pull/242
* adds new mask and saturation filtering when building model in https://github.com/punch-mission/regularizepsf/pull/243

## Version 1.0.2: Nov 16, 2024

- fix broken doc links by @jmbhughes in #228
- Convert str to Path if needed by @jmbhughes in #230

## Version 1.0.1: Nov 2, 2024

- add FITS saving and loading by @jmbhughes in #224
- add visualization for kernels by @jmbhughes in #225

## Version 1.0.0: Nov 2, 2024

- This version moves away from Cython in favor of batching FFTs in SciPy. It's much faster! Plus, you can configure it to run on the GPU (more on that soon). The interface has been completely reworked to a simpler and more elegant solution.
- fix version in docs by @jmbhughes in #221
- Full rewrite for speed and logical simplicity by @jmbhughes in #209

## Version 0.4.2: Nov 1, 2024

- Create .readthedocs.yaml by @jmbhughes in #220

## Version 0.4.1: Oct 30, 2024

- Creates a pinned environment CI by @jmbhughes in #216
- add py3.13 to ci by @jmbhughes in #218
- add py3.13 support with new release by @jmbhughes in #219

## Version 0.4.0: Aug 18, 2024

- Create CITATION.cff by @jmbhughes in #202
- Add variable PSF to variable PSF transforms by @jmbhughes in #203
- bump version by @jmbhughes in #205

## Version 0.3.6: Aug 8, 2024

- This release fixes a mistake in 0.3.5 where the package was not registered properly.

## Version 0.3.5: Aug 7, 2024

- replace requirements_dev.txt in development guide by @jmbhughes in #188
- fix class docstrings and resolve #189 by @jmbhughes in #194
- fix python 3.12 CI failure, enable numpy 2.0 by @jmbhughes in #200

## Version 0.3.4: Jun 18, 2024

- pin versions of numpy by @jmbhughes in #185

## Version 0.3.3: Jun 3, 2024

- Bumps Python version to 3.10 by @jmbhughes in #177

## Version 0.3.2: Apr 11, 2024

- Reverts required Python version to 3.9 instead of 3.10.
- revert-python-bump by @jmbhughes in #165

## Version 0.3.1: Apr 2, 2024

- switching docs to Sphinx by @jmbhughes in #110
- Update docs and readme by @github-actions in #111
- Fix docs build by @jmbhughes in #112
- Fix docs build by @jmbhughes in #113
- Update astropy requirement from ~=5.3 to ~=6.0 by @dependabot in #114
- Revert "Update astropy requirement from ~=5.3 to ~=6.0 (#114)" by @jmbhughes in #117
- Dependabot/pip/astropy approx eq 6.0 by @jmbhughes in #118
- Update astropy requirement from ~=5.3 to ~=6.0 by @dependabot in #119
- Adds pre-commit, updates requirements by @jmbhughes in #121
- Update citations by @jmbhughes in #122
- Finalize pre-commit by @jmbhughes in #123
- Adds and repairs pre-commit, removes pickling of FunctionalCorrector by @github-actions in #124
- [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci in #126
- [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci in #128
- Update pre-commit by @github-actions in #127
- Add paper link by @jmbhughes in #129
- [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci in #130
- [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci in #132
- Updates paper link, updates pre-commit by @github-actions in #131
- [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci in #133
- Update scipy requirement from ~=1.11 to ~=1.12 by @dependabot in #134
- Weekly merge to main by @github-actions in #135
- links to Zenodo for software citation by @jmbhughes in #137
- Updates citation by @github-actions in #138
- [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci in #139
- [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci in #140
- Switch to gnu lgplv3 license by @jmbhughes in #142
- [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci in #143
- update license, create PR template by @jmbhughes in #144
- Update README.md by @jmbhughes in #145
- [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci in #146
- Delete .github/workflows/monthly.yaml by @jmbhughes in #147
- Update ci.yml by @jmbhughes in #148
- [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci in #150
- [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci in #151
- fix link in readme by @jmbhughes in #152
- adds notes about development guide by @jmbhughes in #153
- [pre-commit.ci] pre-commit autoupdate by @pre-commit-ci in #154
- updates release mechanism by @jmbhughes in #155
- updates release mechanism by @jmbhughes in #157


## Version 0.2.3: Nov 2, 2023

### Summary

- Versions of dependencies updates
- Weekly PR automation created
- Matplotlib tests now run properly
- Citation updated

### What's Changed

- Bump cython from 3.0.0 to 3.0.2 by @dependabot in #67
- Bump scikit-image from 0.19.3 to 0.21.0 by @dependabot in #66
- Bump astropy from 5.3.1 to 5.3.3 by @dependabot in #65
- relax version pins to ~= instead of == by @jmbhughes in #77
- adds pytest-mpl to requirements_dev.txt by @jmbhughes in #79
- Update ci.yml by @jmbhughes in #81
- relax version pins, fix mpl tests by @jmbhughes in #78
- Update matplotlib requirement from ~=3.0 to ~=3.8 by @dependabot in #85
- Update scipy requirement from ~=1.10 to ~=1.11 by @dependabot in #84
- Update h5py requirement from ~=3.9 to ~=3.10 by @dependabot in #82
- Update numpy requirement from ~=1.25 to ~=1.26 by @dependabot in #83
- Updates dependency versions by @jmbhughes in #90
- adds code of conduct link by @jmbhughes in #92
- Create weeklypr.yaml by @jmbhughes in #95
- Adds weekly PR, Updates Code of Conduct by @jmbhughes in #96
- Weekly pr fix by @jmbhughes in #97
- Add weekly PR by @jmbhughes in #98
- updates citation, schedules weekly PR by @github-actions in #107
- increment version by @jmbhughes in #108

## Version 0.2.2: Sep 22, 2023

- Updates citation by @jmbhughes in #50
- Create dependabot.yml by @jmbhughes in #51
- Bump lmfit from 1.0.3 to 1.2.2 by @dependabot in #56
- Bump cython from 0.29.32 to 3.0.0 by @dependabot in #55
- Bump astropy from 5.1.1 to 5.3.1 by @dependabot in #54
- Drop deepdish for h5py by @jmbhughes in #59
- Normalize patches by the star-center value rather than the patch maximum by @svank in #60
- increment version by @jmbhughes in #64

## Version 0.2.1: Jul 16, 2023

- Update cite.md by @taniavsn in #33
- Update ci by @jmbhughes in #36
- resolves #39 by @jmbhughes in #40
- docs: fix typo by @sumanchapai in #42
- Update requirements.txt by @jmbhughes in #43
- Create python-publish.yml by @jmbhughes in #44
- Update python-publish.yml by @jmbhughes in #45

## Version 0.2.0: Apr 21, 2023

- This release provides new visualization utilities by @svank. It also fixes some small bugs and improves the speed of model calculation.

- added example code of conduct by @jmbhughes in #7
- Allow passing a custom data loader to find_stars_and_average by @svank in #9
- Wait to pad star cutouts until after averaging by @svank in #15
- Round star coordinates before converting to array coords by @svank in #18
- Avoid numpy dtype deprecation warnings by @svank in #22
- Align interpolation points to data points by @svank in #23
- Add percentile averaging mode by @svank in #20
- Use tmp_path for temp file in test by @svank in #24
- Automatically normalize all PSFs when creating ArrayCorrector by @svank in #28
- Support providing masks for star-finding by @svank in #29
- Take stellar cutouts from the BG-subtracted image by @svank in #19
- updates citation by @jmbhughes in #31
- Visualization utilities by @svank in #17

## Version 0.1.0: Feb 5, 2023

- removes gpu option, adds simulate_observation by @jmbhughes in #4
- fixes major bug when extracting stars and building a model

## Version 0.0.3: Dec 28, 2022

Adds significantly more tests and documentation.

## Version 0.0.2: Dec 2, 2022

Prerelease

## Version 0.0.1: Dec 2, 2022

Prerelease
