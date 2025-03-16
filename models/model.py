import torch
import torch.nn as nn

class PrognosisNN(nn.Module):
    def __init__(self, num_genes, num_diseases, num_chromosomes, num_variants,
                 num_bases, num_aa_props, hidden_dim=128, dropout_prob=0.5):
        super(PrognosisNN, self).__init__()

        # Embedding layers for categorical variables
        self.gene_embed = nn.Embedding(num_genes,128)
        self.disease_embed = nn.Embedding(num_diseases, 128)
        self.chromosome_embed = nn.Embedding(num_chromosomes, 64)
        self.variant_embed = nn.Embedding(num_variants, 16)
        self.base_embed = nn.Embedding(num_bases, 16)

        # Compute the total input dimension dynamically
        embed_dim = 128 + 128 + 64 + 16 + 16 + 16  # Sum of embedding sizes
        numerical_dim = 1 + 1 + 9  # position, zygosity, allele frequency (assuming 9 features)
        biochemical_dim = num_aa_props * 2  # Biochemical properties (assuming num_aa_props=5)
        input_dim = embed_dim + numerical_dim + biochemical_dim

        # Fully connected layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)

        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.01)
        self.sigmoid = nn.Sigmoid()

    def forward(self, gene, disease, chromosome, variant, ref_base, mut_base,
                position, zygosity, allele_freq, aa_orig_props, aa_mut_props):

        # Embed categorical variables
        gene_emb = self.gene_embed(gene)
        disease_emb = self.disease_embed(disease)
        chromosome_emb = self.chromosome_embed(chromosome)
        variant_emb = self.variant_embed(variant)
        ref_base_emb = self.base_embed(ref_base)
        mut_base_emb = self.base_embed(mut_base)

        # Flatten embeddings
        gene_emb = gene_emb.view(gene_emb.size(0), -1)
        disease_emb = disease_emb.view(disease_emb.size(0), -1)
        chromosome_emb = chromosome_emb.view(chromosome_emb.size(0), -1)
        variant_emb = variant_emb.view(variant_emb.size(0), -1)
        ref_base_emb = ref_base_emb.view(ref_base_emb.size(0), -1)
        mut_base_emb = mut_base_emb.view(mut_base_emb.size(0), -1)

        # Ensure numerical features are 2D
        position = position.unsqueeze(1)
        zygosity = zygosity.unsqueeze(1)

        # Concatenate all features
        x = torch.cat([
            gene_emb, disease_emb, chromosome_emb, variant_emb,
            ref_base_emb, mut_base_emb,
            position, zygosity, allele_freq,
            aa_orig_props, aa_mut_props
        ], dim=1)

        # Fully connected layers
        x = self.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))  # Binary classification output

        return x