#!/usr/bin/env python

"""Helper functions for plotly plotting, including choosing samples based on metrics and plotting mutation
and copy number plots."""

from scipy.stats import beta
import pandas as pd
import numpy as np
from intervaltree import IntervalTree
import matplotlib.colors as mcol
import plotly.graph_objects as go
from plotly.subplots import make_subplots

__author__ = "Conor Messer"
__copyright__ = "Copyright 2021, Broad Institute"
__license__ = "BSD-3-Clause"
__version__ = "1.0.1"
__maintainer__ = "Conor Messer"
__email__ = "cmesser@broadinstitute.org"


def choose_samples(metrics_df, blocklist=None, goodlist=None, lt=None, gt=None, best_timing=True,
                   best_qc=None, only_paired=True, best_pre_post=True, separate_pres=False):
    """Choose best samples from metrics dataframe according to inputs.
    
    Blocklist removes these samples from contention; goodlist is a convenience (reciprocal of blocklist) to only use these samples. Next priority is lt/gt, given as dictionaries of {attribute: value} (as sample must be less than or greater than this value for this attribute). If best_pre_post is True, best_timing, and best_qc define how to sort the remaining samples (to choose which are indeed the best pre and post sample. Only one Pre/Post will be given per participant, defined by the given attributes. Finally, if only_paired is True (inferred true if best_pre_post is True), only participants with at least one pre and post sample will be returned (no unmatched pre/post samples).
    
    Returns list of sample names and plot for chosen samples.
    
    Best_qc given as (attribute, bool_ascending)
    """
    metrics_selected = metrics_df.copy()
    if not blocklist:
        blocklist=[]
    if goodlist:
        metrics_selected = metrics_selected.loc[goodlist]
    if not lt:
        lt = {}
    if not gt:
        gt = {}

    metrics_selected.drop(index=blocklist, inplace=True)  # remove blocked samples
    
    # remove samples that don't meet metric thresholds
    for att, val in lt.items():
        metrics_selected = metrics_selected[metrics_selected[att] < val]
    for att, val in gt.items():
        metrics_selected = metrics_selected[metrics_selected[att] > val]
    
    # remove samples that don't have at least one pre and one post
    if only_paired or best_pre_post:
        metrics_selected = remove_non_paired_samples(metrics_selected, separate_pres=separate_pres)
    if best_pre_post:  
        # sort remaining samples according to best_timing/qc - only has an effect if best_pre_post is True
        metrics_selected['dftx_end'] = metrics_selected['dftx_end'].abs()
        if best_timing:
            metrics_selected.sort_values(by=['participant', 'pre_post', 'dftx_start', 'dftx_end'], ascending=[True, True, False, True], inplace=True)
        elif best_qc:
            metrics_selected.sort_values(by=['participant', 'pre_post', best_qc[0]], ascending=[True, True, best_qc[1]], inplace=True)

        if separate_pres:
            metrics_selected.drop_duplicates(subset=['participant', 'pre_post', 'pdb_preservation_method'],
                                             keep='first', inplace=True)
        else:
            metrics_selected.drop_duplicates(subset=['participant', 'pre_post'], keep='first', inplace=True)
    
    return metrics_selected.index.tolist()


def remove_non_paired_samples(sample_df, separate_pres):
    """Remove samples from dataframe that don't have paired pre/post.
    
    Input dataframe should include sample names as index, participant and pre_post as attributes."""
    if separate_pres:
        rna_method = sample_df.groupby(by=['participant', 'pdb_preservation_method', 'pre_post']).size().unstack(
            fill_value=0)
        rna_ff = rna_method.loc[(slice(None), "FF"), :]
        rna_ffpe = rna_method.loc[(slice(None), "FFPE"), :]
        rna_ff_p = rna_ff[(rna_ff['Post'] > 0) & (rna_ff['Pre'] > 0)].reset_index()['participant'].values
        rna_ffpe_p = rna_ffpe[(rna_ffpe['Post'] > 0) & (rna_ffpe['Pre'] > 0)].reset_index()['participant'].values
        return sample_df[((sample_df['participant'].isin(rna_ff_p)) & (sample_df['pdb_preservation_method'] == 'FF')) |
                         ((sample_df['participant'].isin(rna_ffpe_p)) & (sample_df['pdb_preservation_method'] == 'FFPE'))]
    else:
        pre_post_rna_p = set(sample_df[sample_df['pre_post'] == 'Pre']['participant'].unique()) & \
                         set(sample_df[sample_df['pre_post'] == 'Post']['participant'].unique())
        return sample_df[sample_df['participant'].isin(pre_post_rna_p)].copy()

    
def get_seg_tree(seg_dfs, seg_cluster_df):
    """Make a tree for the segments of this participant given by these seg

    :param seg_files: dict with Sample_ID:seg_file_df
    :return: list of IntervalTree givings segs for each chromosome
    """
    # get all capseg files for this participant
    sample_names = list(seg_dfs.keys())
    seg_cluster_df = seg_cluster_df.astype({'Start.bp': int, 'End.bp': int})
    contig_trees = []

    for contig in pd.unique(seg_dfs[sample_names[0]]['Chromosome']):
        tree1 = IntervalTree()
        for i, s_name in enumerate(sample_names):

            this_seg_df = seg_dfs[s_name]

            contig_seg_df = this_seg_df.loc[this_seg_df['Chromosome'] == contig]
            start_bps = contig_seg_df['Start.bp'].tolist()
            end_bps = contig_seg_df['End.bp'].tolist()
            hscr_a1s = contig_seg_df['mu.minor'].tolist()
            hscr_a2s = contig_seg_df['mu.major'].tolist()
            sigma_1 = contig_seg_df['sigma.minor'].tolist()
            sigma_2 = contig_seg_df['sigma.major'].tolist()

            for j in range(len(start_bps)):
                tree1.addi(start_bps[j], end_bps[j], {s_name: [hscr_a1s[j], hscr_a2s[j],
                                                               sigma_1[j], sigma_2[j]]})

        tree1.split_overlaps()
        tree1.merge_equals(data_reducer=reducer)

        # make tree for this chromosome from phylogic_seg_cluster file
        this_chrom_cluster = seg_cluster_df[seg_cluster_df['Chromosome'] == contig]
        cluster_tree = IntervalTree.from_tuples([(s, e, d) for s, e, d in zip(this_chrom_cluster['Start.bp'],
                                                                              this_chrom_cluster['End.bp'],
                                                                              this_chrom_cluster['Cluster_assignment'])])
        # need to test to make sure only one cluster given for each segment (and what to do if none given***)
        tree_with_clusters = []
        for interval_obj in tree1:
            cluster_tree_segs = cluster_tree.overlap(interval_obj.begin, interval_obj.end)
            if len(cluster_tree_segs) > 1:
                raise ValueError(f'MORE THAN ONE CLUSTER in interval {interval_obj.begin} - {interval_obj.end}')
            elif not cluster_tree_segs:   # empty set
                single_cluster = 0
            else:
                single_cluster = cluster_tree_segs.pop().data
            # append cluster onto the data list for each sample in this interval
            # (trying to mutate data directly leads to bugs)
            data = {sample: old_data + [single_cluster] for sample, old_data in interval_obj.data.items()}
            tree_with_clusters.append((interval_obj.begin, interval_obj.end, data))

        contig_trees.append(IntervalTree.from_tuples(tree_with_clusters))

    return contig_trees


def reducer(old, new):
    return dict(old, **new)


def get_tree_data(all_seg_trees, chrom, pos, sample, i):
    """Returns data at given chromosome, position, and sample with 'NA - no seg' returned if data doesn't exist."""
    try:
        seg_data = list(all_seg_trees[chrom - 1][pos])[0].data[sample][i] # only one hit, because of split_overlaps
    except IndexError:
        seg_data = 'NA - no seg'
    except KeyError:
        seg_data = 'NA - no seg'
    finally:
        return seg_data


def calculate_error(alt, ref, purity, percentile):
    """Calculates error for mutation based on beta distribution for given alt and ref read counts and purity."""
    if alt == 0:
        return 0
    else:
        return (beta.ppf(percentile, alt, ref) - alt / (alt + ref)) / purity
    
    
def make_mut_seg_plot(mut_df, seg_trees, sample_order, cr_diff_dict, c0_dict):
    """Make an allelic copy ratio plot with mutations overlaid on the segments."""
    # pass in as variable?
    c_size = [249250621, 243199373, 198022430, 191154276, 180915260, 171115067, 159138663, 146364022, 141213431,
              135534747, 135006516, 133851895, 115169878, 107349540, 102531392, 90354753, 81195210, 78077248, 59128983,
              63025520, 48129895, 51304566, 156040895, 57227415]  # Chromosome sizes

    sample_names = sorted(sample_order, key=lambda k: int(sample_order[k]))

    chroms = list(range(1, 24))
    base_start = 0
    dy = 0.07
    chrom_ticks = []
    patch_color = 'gainsboro'
    patch_list = [patch_color]
    seg_diff_cmap = mcol.LinearSegmentedColormap.from_list("Blue_Red", ["b", "r"], 100)
    phylogic_color_dict = get_phylogic_color_scale()

    c_size_cumsum = np.cumsum([0] + c_size)
    mut_df['x_loc'] = mut_df.apply(lambda x: calc_x_mut(x.Start_position, x.Chromosome, c_size_cumsum), axis=1)
    mut_df['cluster_color'] = mut_df['Cluster_Assignment'].apply(lambda x: phylogic_color_dict[x])

    # make subplots - to accommodate shapes and possible extension
    fig = make_subplots(len(sample_names), 1, shared_xaxes=True,
                        vertical_spacing=0.03, row_heights=[350]*len(sample_names),
                        subplot_titles=sample_names)

    fig.update_yaxes(range=[-1, 6])
    fig.update_traces(marker_line_width=0)

    seg_list = []
    for i in range(len(seg_trees)):
        for segment in seg_trees[i]:
            if segment[1] - segment[0] > 50000:  # Don't plot extremely short segs
                for j, sample in enumerate(sample_names):
                    try:
                        mu_minor = segment[2].get(sample)[0]  # use function?
                        if np.isnan(mu_minor):
                            raise TypeError
                    except TypeError:
                        pass
                        # print(f"Segment {segment[0]} to {segment[1]} on chr {i + 1} doesn't exist for {sample_name}.")
                    else:
                        mu_major = segment[2].get(sample)[1]
                        sigma_minor = segment[2].get(sample)[2]
                        sigma_major = segment[2].get(sample)[3]
                        cluster = segment[2].get(sample)[4]
                        cluster_color = phylogic_color_dict[str(cluster)]

                        mu_diff = mu_major - mu_minor
                        maj_val = int(np.ceil(50 + 50 * calc_color(mu_diff)))
                        min_val = int(np.floor(50 - 50 * calc_color(mu_diff)))

                        # get adjusted CN values
                        mu_major_adj = (mu_major - c0_dict[sample]) / cr_diff_dict[sample]
                        mu_minor_adj = (mu_minor - c0_dict[sample]) / cr_diff_dict[sample]
                        sigma_major_adj = sigma_major / cr_diff_dict[sample]
                        sigma_minor_adj = sigma_minor / cr_diff_dict[sample]

                        # row number defined by sample order
                        row_num = j+1

                        start = segment.begin + base_start
                        end = segment.end + base_start

                        seg_list.append([i+1, start, end, cluster, cluster_color,
                                         mu_major, mu_minor, sigma_major, sigma_minor,
                                         mu_major_adj, mu_minor_adj, sigma_major_adj, sigma_minor_adj,
                                         maj_val, min_val, sample, row_num])

        patch_color = 'white' if patch_color == 'gainsboro' else 'gainsboro'  # Alternate background color between chromosomes
        patch_list.append(patch_color)
        chrom_ticks.append(base_start + c_size[i] / 2)

        base_start += c_size[i]

    seg_df = pd.DataFrame(seg_list,
                          columns=['Chromosome', 'Start_pos', 'End_pos', 'Cluster_Assignment', 'cluster_color',
                                   'mu_major', 'mu_minor', 'sigma_major', 'sigma_minor',
                                   'mu_major_adj', 'mu_minor_adj', 'sigma_major_adj', 'sigma_minor_adj',
                                   'maj_diff', 'min_diff', 'Sample_ID', 'row_num'])
    seg_df = seg_df.sort_values(['Sample_ID'], key=lambda x: x.map(sample_order))
    # can move seg traces out of for loop now?
    #  Keep track of traces of segments?
    trace_nums = {}
    trace_counter = 0
    for row, sample in enumerate(sample_names):
        this_sample_seg = seg_df[seg_df['Sample_ID'] == sample]
        this_sample_seg.apply(lambda x: make_cnv_scatter(fig, x.Start_pos, x.End_pos, x.mu_major_adj, x.mu_minor_adj,
                                                         dy, x.cluster_color, row+1), axis=1)
        new_counter = trace_counter + 2 * len(this_sample_seg)
        trace_nums[sample] = (trace_counter, new_counter)  # keeps track of trace numbers for this sample's segments

        fig.add_trace(make_mut_scatter(mut_df[mut_df['Sample_ID'] == sample]), row=row+1, col=1)
        trace_counter = new_counter + 1

    # flip order of data so mutations are plotted last (on top)
    # fig.data = fig.data[::-1]

    # add chromosome lines/rectangles
    for i in range(len(seg_trees)):
        fig.add_vrect(c_size_cumsum[i], c_size_cumsum[i + 1], fillcolor=patch_list[i],
                      opacity=1, layer='below', line_width=0)

    y_ticks = np.arange(0, 5, 1)

    # Draw lines at absolute copy numbers ??

    # modify layout
    fig.update_xaxes(showgrid=False,
                     zeroline=False,
                     tickvals=chrom_ticks,
                     ticktext=chroms,  # fontsize=6,
                     tickangle=0,
                     range=[0, base_start])
    fig.update_xaxes(title_text="Chromosome", row=len(sample_names), col=1)
    fig.update_yaxes(showgrid=False,
                     zeroline=False,
                     tickvals=y_ticks,
                     ticktext=list(range(6)),  # fontsize=16
                     ticks="outside",
                     range=[-0.5, 4.5],
                     title_text="Copy Number")

    ################
    fig.update_layout(title=mut_df.iloc[0]['Patient_ID'])

    return fig, seg_df, trace_nums


def calc_multiplicity(mut_series, purity, cr_diff, c0):
    """Calculate multiplicity for the mutation"""
    mu_min_adj = (mut_series['mu_minor'] - c0) / cr_diff
    mu_maj_adj = (mut_series['mu_major'] - c0) / cr_diff

    # returns the multiplicity * CCF for this mutation
    return mut_series['VAF'] * (purity * (mu_min_adj + mu_maj_adj) + 2 * (1 - purity)) / purity


def calc_x_mut(pos, chrom, chrom_sizes):
    return chrom_sizes[chrom - 1] + pos


def calc_color(mu_diff):
    return (7*mu_diff**2) / (7*mu_diff**2 + 10)


def get_rgb_string(c):
    return '({},{},{})'.format(*c)


def get_hex_string(c):
    return '#{:02X}{:02X}{:02X}'.format(*c)


def get_phylogic_color_scale():
    phylogic_color_list = [[166, 17, 129],
                           [39, 140, 24],
                           [103, 200, 243],
                           [248, 139, 16],
                           [16, 49, 41],
                           [93, 119, 254],
                           [152, 22, 26],
                           [104, 236, 172],
                           [249, 142, 135],
                           [55, 18, 48],
                           [83, 82, 22],
                           [247, 36, 36],
                           [0, 79, 114],
                           [243, 65, 132],
                           [60, 185, 179],
                           [185, 177, 243],
                           [139, 34, 67],
                           [178, 41, 186],
                           [58, 146, 231],
                           [130, 159, 21],
                           [161, 91, 243],
                           [131, 61, 17],
                           [248, 75, 81]]
    colors_dict = {str(i): get_hex_string(c) for i, c in enumerate(phylogic_color_list)}
    return colors_dict

def make_mut_scatter(mut_df):
    """Create a scatter plot with all mutations in the dataframe.

    Not using plotly express because it returns a separate trace for each color (each cluster).
    """
    mut_scatter = go.Scatter(x=mut_df['x_loc'], y=mut_df['multiplicity_ccf'],
                               mode='markers', marker_size=10,
                               marker_color=mut_df['cluster_color'],
                               error_y=dict(type='data',
                                            array=mut_df['error_top'],
                                            arrayminus=mut_df['error_bottom'],
                                            color='gray',
                                            visible=True,
                                            width=0),
                               customdata=np.stack((mut_df['Hugo_Symbol'].tolist(),
                                                    mut_df['Chromosome'].tolist(),
                                                    mut_df['Start_position'].tolist(),
                                                    mut_df['VAF'].tolist(),
                                                    mut_df['Cluster_Assignment'].tolist(),
                                                    mut_df['Variant_Type'].tolist(),
                                                    mut_df['Variant_Classification'].tolist(),
                                                    mut_df['Protein_change']),
                                                   axis=-1),
                               hovertemplate='<extra></extra>' +
                                             'Gene: %{customdata[0]} %{customdata[1]}:%{customdata[2]} <br>' +
                                             'Variant: %{customdata[5]}, %{customdata[6]} <br>' +
                                             'Protein Change: %{customdata[7]} <br>' +
                                             'Multiplicity: %{y:.3f} <br>' +
                                             'VAF: %{customdata[3]:.3f} <br>' +
                                             'Cluster: %{customdata[4]:d}')
    return mut_scatter


def make_cnv_scatter(fig, start, end, mu_maj_adj, mu_min_adj, dy, color, row):
    """Make a scatter plot for each of the minor and major alleles as filled rectangles."""
    fig.add_trace(go.Scatter(x=[start, start, end, end],
                      y=[mu_maj_adj + dy / 2, mu_maj_adj - dy / 2, mu_maj_adj - dy / 2, mu_maj_adj + dy / 2],
                      fill='toself', fillcolor=color, mode='none',
                      hoverinfo='none',
                      showlegend=False), row=row, col=1)
    fig.add_trace(go.Scatter(x=[start, start, end, end],
                      y=[mu_min_adj + dy / 2, mu_min_adj - dy / 2, mu_min_adj - dy / 2, mu_min_adj + dy / 2],
                      fill='toself', fillcolor=color, mode='none',
                      hoverinfo='none',
                      showlegend=False), row=row, col=1)