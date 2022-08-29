use crate::meanshift_base::LibData;

pub fn dtw<A: LibData>(a: &[A], b: &[A]) -> A {
  let mut cost_matrix = vec![vec![A::max_value(); b.len()]; a.len()];
  cost_matrix[0][0] = A::from(0.).unwrap();

  for i in 1..a.len() {
    for j in 1..b.len() {
      let cost = a[i - 1] - b[j - 1];
      cost_matrix[i][j] = cost + A::min(
        A::min(
          cost_matrix[i - 1][j],
          cost_matrix[i][j - 1]
        ),
        cost_matrix[i - 1][j - 1]
      );
    }
  }

  return cost_matrix[a.len()][b.len()];
}
