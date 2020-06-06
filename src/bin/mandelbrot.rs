// The Computer Language Benchmarks Game
// https://salsa.debian.org/benchmarksgame-team/benchmarksgame/
//
// Contributed by Prathmesh Prabhu
// based on the C program contributed by Kevin Miller
// inspired by the C-to-Rust conversion described at
// http://cliffle.com/p/dangerust/

// Will outlast current iteration.
#![allow(non_upper_case_globals, non_camel_case_types, non_snake_case)]

use itertools::iproduct;
use rayon::prelude::*;
use std::arch::x86_64::__m128d;
use std::io::Write;

#[cfg(target_feature = "sse2")]
mod mm {
    use std::arch::x86_64::*;

    pub fn extract_lower(v: __m128d) -> f64 {
        // Safety: Only compiled when the target supports SSE2 instructions.
        unsafe { _mm_cvtsd_f64(_mm_shuffle_pd(v, v, 0b0000_00000)) }
    }

    pub fn extract_upper(v: __m128d) -> f64 {
        // Safety: Only compiled when the target supports SSE2 instructions.
        unsafe { _mm_cvtsd_f64(_mm_shuffle_pd(v, v, 0b0000_00001)) }
    }

    pub fn mul_pd(a: __m128d, b: __m128d) -> __m128d {
        // Safety: Only compiled when the target supports SSE2 instructions.
        unsafe { _mm_mul_pd(a, b) }
    }

    pub fn div_pd(a: __m128d, b: __m128d) -> __m128d {
        // Safety: Only compiled when the target supports SSE2 instructions.
        unsafe { _mm_div_pd(a, b) }
    }

    pub fn add_pd(a: __m128d, b: __m128d) -> __m128d {
        // Safety: Only compiled when the target supports SSE2 instructions.
        unsafe { _mm_add_pd(a, b) }
    }

    pub fn sub_pd(a: __m128d, b: __m128d) -> __m128d {
        // Safety: Only compiled when the target supports SSE2 instructions.
        unsafe { _mm_sub_pd(a, b) }
    }

    pub fn set_pd(a: f64, b: f64) -> __m128d {
        // Safety: Only compiled when the target supports SSE2 instructions.
        unsafe { _mm_set_pd(a, b) }
    }

    pub fn set1_pd(a: f64) -> __m128d {
        // Safety: Only compiled when the target supports SSE2 instructions.
        unsafe { _mm_set1_pd(a) }
    }
}

fn vec_nle(v: &[__m128d; 4], f: f64) -> bool {
    // https://github.com/searchivarius/BlogCode/blob/master/2014/5/14/mm_extract_pd.cpp
    // is "more correct" and sometimes faster than the original C programs
    // indexing of f64 values within an __m128d.

    // TODO: Also see gcc option -mfpmath=sse which would lead to normal
    // floating point operations also using SSE registers. This could lead to
    // more efficient type punning.
    // https://stackoverflow.com/questions/12624466/get-member-of-m128-by-index

    // Rust compiler had trouble unrolling loop here, which led to a noticeable
    // performance hit, so unroll manually.
    mm::extract_lower(v[0]) > f
        && mm::extract_upper(v[0]) > f
        && mm::extract_lower(v[1]) > f
        && mm::extract_upper(v[1]) > f
        && mm::extract_lower(v[2]) > f
        && mm::extract_upper(v[2]) > f
        && mm::extract_lower(v[3]) > f
        && mm::extract_upper(v[3]) > f
}

fn clrPixels_nle(v: &[__m128d; 4], f: f64, pix8: &mut u8) {
    if !(mm::extract_lower(v[0]) <= f) {
        *pix8 &= 0b0111_1111; // 0x7f
    }
    if !(mm::extract_upper(v[0]) <= f) {
        *pix8 &= 0b1011_1111; // 0xbf
    }
    if !(mm::extract_lower(v[1]) <= f) {
        *pix8 &= 0b1101_1111; // 0xdf
    }
    if !(mm::extract_upper(v[1]) <= f) {
        *pix8 &= 0b1110_1111; // 0xef
    }
    if !(mm::extract_lower(v[2]) <= f) {
        *pix8 &= 0b1111_0111; // 0xf7
    }
    if !(mm::extract_upper(v[2]) <= f) {
        *pix8 &= 0b1111_1011; // 0xfb
    }
    if !(mm::extract_lower(v[3]) <= f) {
        *pix8 &= 0b1111_1101; // 0xfd
    }
    if !(mm::extract_upper(v[3]) <= f) {
        *pix8 &= 0b1111_1110; // 0xfe
    }
}

fn calcSum(
    r: &mut [__m128d; 4],
    i: &mut [__m128d; 4],
    sum: &mut [__m128d; 4],
    init_r: &[__m128d; 4],
    init_i: __m128d,
) {
    for idx in 0..4 {
        // Create local variables to avoid overly broad unsafe blocks.
        let cur_r = r[idx];
        let cur_i = i[idx];
        let cur_init_r = init_r[idx];

        let r2: __m128d = mm::mul_pd(cur_r, cur_r);
        let i2: __m128d = mm::mul_pd(cur_i, cur_i);
        let ri: __m128d = mm::mul_pd(cur_r, cur_i);
        sum[idx] = mm::add_pd(r2, i2);
        r[idx] = mm::add_pd(mm::sub_pd(r2, i2), cur_init_r);
        i[idx] = mm::add_pd(mm::add_pd(ri, ri), init_i);
    }
}

fn mand8(init_r: &[__m128d; 4], init_i: __m128d) -> u8 {
    let mut r = *init_r;
    let mut i = [init_i, init_i, init_i, init_i];
    let zero = mm::set1_pd(0.0);
    let mut sum = [zero; 4];

    let mut pix8: u8 = 0xff;
    for _ in 0..6 {
        for _ in 0..8 {
            calcSum(&mut r, &mut i, &mut sum, &init_r, init_i);
        }

        if vec_nle(&sum, 4.0) {
            pix8 = 0x00;
            break;
        }
    }
    if pix8 != 0 {
        calcSum(&mut r, &mut i, &mut sum, &init_r, init_i);
        calcSum(&mut r, &mut i, &mut sum, &init_r, init_i);
        clrPixels_nle(&sum, 4.0, &mut pix8);
    }
    return pix8;
}

fn calc_init_r_pair(x: f64, wid_ht: f64) -> __m128d {
    mm::sub_pd(
        mm::mul_pd(
            // NB: mm::set_pd() reverses the order of arguments when packing
            // f64 into a __m128d.
            // https://stackoverflow.com/questions/5237961/why-does-does-sse-set-mm-set-ps-reverse-the-order-of-arguments
            mm::set_pd(x + 1.0, x),
            mm::div_pd(mm::set1_pd(2.0), mm::set1_pd(wid_ht)),
        ),
        mm::set1_pd(1.5),
    )
}

fn calc_init_r_chunk(x: f64, wid_ht: f64) -> [__m128d; 4] {
    [
        calc_init_r_pair(x as f64, wid_ht as f64),
        calc_init_r_pair((x + 2.0) as f64, wid_ht as f64),
        calc_init_r_pair((x + 4.0) as f64, wid_ht as f64),
        calc_init_r_pair((x + 6.0) as f64, wid_ht as f64),
    ]
}

fn main() {
    // get width/height from arguments
    let mut wid_ht: usize = 16000;
    if !std::env::args().len() > 0 {
        wid_ht = std::env::args().nth(1).unwrap().parse().unwrap();
    }
    wid_ht = (wid_ht + 7) & !(7 as usize);

    std::io::stdout()
        .write_all(format!("P4\n{} {}\n", wid_ht, wid_ht).as_bytes())
        .unwrap();

    let mut pixels: Vec<u8> = Vec::with_capacity(wid_ht * (wid_ht >> 3));

    let i0 = {
        let mut i0: Vec<__m128d> = Vec::with_capacity(wid_ht);
        for y in 0..wid_ht {
            let v = (2.0 / wid_ht as f64) * y as f64 - 1.0;
            i0.push(mm::set_pd(v, v));
        }
        i0
    };

    // process 8 pixels (one byte) at a time
    let r0 = {
        let mut r0: Vec<[__m128d; 4]> = Vec::with_capacity(wid_ht / 8);
        for x in (0..wid_ht).step_by(8) {
            r0.push(calc_init_r_chunk(x as f64, wid_ht as f64));
        }
        r0
    };

    if std::env::var("PARALLELIZE").is_ok() {
        parallel::new(&i0, &r0)
            .map(|(i, r)| mand8(r, i))
            .collect_into_vec(&mut pixels);
    } else {
        pixels.extend(iproduct!(i0.iter(), r0.iter()).map(|(i, r)| mand8(&*r, *i)));
    }
    std::io::stdout().write_all(pixels.as_slice()).unwrap();
}

mod parallel {
    use rayon::iter::plumbing::{bridge, Consumer, ProducerCallback, UnindexedConsumer};
    use rayon::prelude::*;
    use std::arch::x86_64::__m128d;

    pub fn new<'iter>(i0: &'iter Vec<__m128d>, r0: &'iter Vec<[__m128d; 4]>) -> GridIter<'iter> {
        GridIter { i0, r0 }
    }

    pub struct GridIter<'iter> {
        i0: &'iter Vec<__m128d>,
        r0: &'iter Vec<[__m128d; 4]>,
    }

    impl<'iter> ParallelIterator for GridIter<'iter> {
        type Item = (__m128d, &'iter [__m128d; 4]);
        fn drive_unindexed<C>(self, consumer: C) -> C::Result
        where
            C: UnindexedConsumer<Self::Item>,
        {
            bridge(self, consumer)
        }

        fn opt_len(&self) -> Option<usize> {
            Some(self.i0.len() * self.r0.len())
        }
    }

    impl<'iter> IndexedParallelIterator for GridIter<'iter> {
        fn drive<C>(self, consumer: C) -> C::Result
        where
            C: Consumer<Self::Item>,
        {
            bridge(self, consumer)
        }

        fn len(&self) -> usize {
            self.i0.len() * self.r0.len()
        }

        fn with_producer<CB>(self, callback: CB) -> CB::Output
        where
            CB: ProducerCallback<Self::Item>,
        {
            callback.callback(GridProducer {
                i0: self.i0,
                r0: self.r0,
                left: 0,
                right: self.len(),
            })
        }
    }

    struct GridProducer<'producer> {
        i0: &'producer Vec<__m128d>,
        r0: &'producer Vec<[__m128d; 4]>,
        left: usize,
        right: usize,
    }

    impl<'producer> rayon::iter::plumbing::Producer for GridProducer<'producer> {
        fn split_at(self, index: usize) -> (Self, Self) {
            (
                GridProducer {
                    i0: self.i0,
                    r0: self.r0,
                    left: self.left,
                    right: index,
                },
                GridProducer {
                    i0: self.i0,
                    r0: self.r0,
                    left: index,
                    right: self.right,
                },
            )
        }

        type Item = (__m128d, &'producer [__m128d; 4]);
        type IntoIter = GridPartIter<'producer>;

        fn into_iter(self) -> Self::IntoIter {
            // This probably incurs a bunch of bounds checks that are worth
            // skipping (?) with unsafe code.
            let len = self.right - self.left;
            let row_len = self.r0.len();
            // Consider modifying self.left directly instead of the local var.
            let mut left = self.left;
            // Consider modifying self.right directly instead of the local var.
            let mut remaining = len;
            let head = RowPartIter {
                i: self.i0.get(left / row_len).unwrap(),
                rs: self.r0[left % row_len..std::cmp::min(row_len - left % row_len, remaining)]
                    .iter(),
            };
            remaining -= head.rs.len();
            left += head.rs.len();

            // We expect rest to be empty most of the time.
            // When it _is_ populated, we expect the length to be 1.
            let mut rest = vec![];
            while remaining > 0 {
                let next = RowPartIter {
                    i: self.i0.get(left / row_len).unwrap(),
                    rs: self.r0[left % row_len..std::cmp::min(row_len - left % row_len, remaining)]
                        .iter(),
                };
                remaining -= head.rs.len();
                left += head.rs.len();
                rest.push(next);
            }

            GridPartIter {
                head,
                rest: rest.into_iter(),
                tail: None,
                len,
            }
        }
    }

    struct GridPartIter<'iter> {
        head: RowPartIter<'iter>,
        rest: std::vec::IntoIter<RowPartIter<'iter>>,
        tail: Option<RowPartIter<'iter>>,
        len: usize,
    }

    impl<'iter> Iterator for GridPartIter<'iter> {
        type Item = (__m128d, &'iter [__m128d; 4]);

        fn next(&mut self) -> Option<Self::Item> {
            if let Some(e) = self.head.next() {
                return Some(e);
            }
            if let Some(head) = self.rest.next() {
                self.head = head;
                return self.next();
            }
            let mut tail = None;
            std::mem::swap(&mut self.tail, &mut tail);
            if let Some(head) = tail {
                self.head = head;
                return self.next();
            }
            return None;
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            (self.len, Some(self.len))
        }
    }

    impl<'iter> ExactSizeIterator for GridPartIter<'iter> {
        fn len(&self) -> usize {
            self.len
        }
    }

    impl<'iter> DoubleEndedIterator for GridPartIter<'iter> {
        fn next_back(&mut self) -> Option<Self::Item> {
            if let Some(tail) = &mut self.tail {
                if let Some(e) = tail.next_back() {
                    return Some(e);
                }
            }
            if let Some(tail) = self.rest.next_back() {
                self.tail = Some(tail);
                return self.next_back();
            }
            self.head.next_back()
        }
    }

    struct RowPartIter<'iter> {
        i: &'iter __m128d,
        rs: std::slice::Iter<'iter, [__m128d; 4]>,
    }

    impl<'iter> Iterator for RowPartIter<'iter> {
        type Item = (__m128d, &'iter [__m128d; 4]);

        fn next(&mut self) -> Option<Self::Item> {
            self.rs.next().map(|r| (*self.i, r))
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            self.rs.size_hint()
        }
    }

    impl<'iter> ExactSizeIterator for RowPartIter<'iter> {
        fn len(&self) -> usize {
            self.rs.len()
        }
    }

    impl<'iter> DoubleEndedIterator for RowPartIter<'iter> {
        fn next_back(&mut self) -> Option<Self::Item> {
            self.rs.next_back().map(|r| (*self.i, r))
        }
    }
}
