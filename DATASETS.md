# Dataset notices

The root [MIT license](LICENSE) covers the DataLens software and documentation.
It does **not** relicense the files in `sample_data/`. Each fixture keeps the
license and attribution declared by its original publisher.

## Netflix Movies and TV Shows

- Files: `sample_data/netflix_titles.csv`
- Source: [Netflix Movies and TV Shows on Kaggle](https://www.kaggle.com/datasets/shivamb/netflix-shows)
- Publisher: Shivam Bansal (`shivamb` on Kaggle)
- Source-declared license: [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/)
- Included license text: [`sample_data/LICENSES/CC0-1.0.txt`](sample_data/LICENSES/CC0-1.0.txt)
- Purpose here: insight-first profiling and regression testing

## Customer Churn Dataset

- Files: `sample_data/customer_churn_dataset-training-master.csv` and
  `sample_data/customer_churn_dataset-testing-master.csv`
- Source: [Customer Churn Dataset on Kaggle](https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset)
- Publisher: Muhammad Shahid Azeem
- Source-declared license: [GNU General Public License v2.0](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)
- Included license text: [`sample_data/LICENSES/GPL-2.0-only.txt`](sample_data/LICENSES/GPL-2.0-only.txt)
- Purpose here: supervised-model and external-validation regression testing

The source pages are the authority for dataset terms. Review them before
redistributing a fixture or using it outside this repository. Dataset contents
are not endorsed by, or affiliated with, Netflix or the individuals represented
in the source data. The Docker image includes this notice, the root MIT license,
and the dataset license texts alongside the bundled fixtures.

## Fixture integrity

Exact hashes, schemas, and acceptance facts are recorded in
[`sample_data/fixtures.json`](sample_data/fixtures.json). A fixture replacement
must update its file, provenance, license record, schema, hash, and expected
facts in the same pull request.
